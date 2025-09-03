# src/musicgen_ext/modeling_musicgen_ext.py
import torch
from transformers import MusicgenForConditionalGeneration
from typing import Optional

class MusicgenWithContext(MusicgenForConditionalGeneration):
    """
    Drop-in replacement that lets you
        (1) cache_past_tokens(prompt_ids)
        (2) generate_continuation(new_ids, …)
    """
class MusicgenWithContext(MusicgenForConditionalGeneration):
    """
    1. cache_past_tokens(prompt_ids)  – primes self._pkv with a prompt.
    2. generate_continuation(...)     – continues from *external* KV cache.
    """

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def cache_past_tokens(self, input_ids, **gen_kwargs):
        gen_kwargs.setdefault("return_dict_in_generate", True)
        gen_kwargs.setdefault("max_new_tokens", 1)   # generate 1 frame
        gen_kwargs.setdefault("do_sample", False)

        out          = super().generate(input_ids, **gen_kwargs)
        self._pkv    = out.past_key_values          # full prompt KV
        return self._pkv

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_continuation(
        self,
        input_ids:       torch.LongTensor | None = None,
        inputs_embeds:   torch.FloatTensor | None = None,
        max_new_tokens:  int = 100,
        streamer=None,
        **gen_kwargs,
    ):
        if not hasattr(self, "_pkv"):
            raise RuntimeError("Call cache_past_tokens() once first.")

        # --------------------------------------------------------------
        # 1) take *caller-supplied* KV (rolling window) if provided
        # --------------------------------------------------------------
        ext_pkv = gen_kwargs.pop("past_key_values", None)
        if ext_pkv is not None:
            self._pkv = ext_pkv                     # <-- overwrite

        # keep return type predictable
        gen_kwargs.setdefault("return_dict_in_generate", True)

        # --------------------------------------------------------------
        # 2) build the minimal input dict
        # --------------------------------------------------------------
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds`.")

        model_inputs = {"input_ids": input_ids} if input_ids is not None else \
                       {"inputs_embeds": inputs_embeds}

        # --------------------------------------------------------------
        # 3) run generation
        # --------------------------------------------------------------
        out = super().generate(
            past_key_values = self._pkv,
            use_cache       = True,
            max_new_tokens  = int(max_new_tokens),
            streamer        = streamer,
            **model_inputs,
            **gen_kwargs,
        )

        # --------------------------------------------------------------
        # 4) advance internal pointer so next call uses newest KV
        #    (caller probably appends to RollingKVCache too)
        # --------------------------------------------------------------
        self._pkv = out.past_key_values
        return out

# # Testing with token crossfade
# # ------------------------------------------------------------
# class MusicgenWithContextAndFade(MusicgenWithContext):
#     def generate_continuation(self,
#                               input_ids=None,
#                               alpha_len_frames: int = 8,   # ≈160 ms
#                               **kwargs):

#         # 1) cache old pkv so we can mix later
#         pkv_old = self._pkv
#         super_out = super().generate(input_ids=input_ids,
#                                      **kwargs,
#                                      output_hidden_states=True,
#                                      return_dict_in_generate=True)

#         hidden_new = super_out.decoder_hidden_states  # tuple(layer) len=L
#         with torch.no_grad():
#             mixed_hidden = []
#             for l, h in enumerate(hidden_new):
#                 # h : [B, T, D] where T = frames*4 tokens
#                 if h.size(1) < alpha_len_frames*4:
#                     mixed_hidden.append(h)     # too short – skip
#                     continue

#                 # alpha ramp over first alpha_len_frames*4 tokens
#                 Tmix = alpha_len_frames*4
#                 alpha = torch.linspace(0, 1, Tmix, device=h.device)  # 0→1
#                 alpha = alpha.view(1, Tmix, 1)

#                 # old hidden state = last pkv_old K projected back (cheap hack)
#                 # we approximate with first tokens of old prompt rep:
#                 h_old = h[:, :Tmix, :].clone()   # same shape
#                 h[:, :Tmix, :] = (1-alpha)*h_old + alpha*h[:, :Tmix, :]
#                 mixed_hidden.append(h)

#             # replace hidden states (only first layer needed for logits)
#             super_out.decoder_hidden_states = tuple(mixed_hidden)

#         return super_out
