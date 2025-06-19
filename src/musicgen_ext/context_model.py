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
    @torch.inference_mode()
    def cache_past_tokens(self, input_ids, **gen_kwargs):

        # --------------------------------------------------------------- #
        # make sure generate() yields a *dict* with .past_key_values
        gen_kwargs.setdefault("return_dict_in_generate", True)
        gen_kwargs.setdefault("max_new_tokens", 1)   # any positive value
        gen_kwargs.setdefault("do_sample", False)
        # --------------------------------------------------------------- #

        out = super().generate(input_ids, **gen_kwargs)
        self._pkv = out.past_key_values
        return self._pkv

    @torch.inference_mode()
    def generate_continuation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 100,
        streamer=None,
        **gen_kwargs,
    ):
        if not hasattr(self, "_pkv"):
            raise RuntimeError("Call cache_past_tokens() first.")

        # drop any duplicate kwarg coming from the caller
        gen_kwargs.pop("past_key_values", None)

        # ------------------------------------------------------------------
        # Build a kw-dict that contains **exactly one** of the two fields
        # ------------------------------------------------------------------
        model_inputs = {}
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        elif inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be given.")

        def _as_int(x, name):
            if isinstance(x, float):
                if not x.is_integer():
                    raise ValueError(f"Warning {name} must be an integer, got {x}")
                x = int(x)
            return x
        
        # make sure we get a dict back
        gen_kwargs.setdefault("return_dict_in_generate", True)
        
        max_new_tokens = _as_int(max_new_tokens, "max_new_tokens")

        return super().generate(
            past_key_values=self._pkv,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            **model_inputs,
            **gen_kwargs,
        )


# Testing with token crossfade
# ------------------------------------------------------------
class MusicgenWithContextAndFade(MusicgenWithContext):
    def generate_continuation(self,
                              input_ids=None,
                              alpha_len_frames: int = 8,   # ≈160 ms
                              **kwargs):

        # 1) cache old pkv so we can mix later
        pkv_old = self._pkv
        super_out = super().generate(input_ids=input_ids,
                                     **kwargs,
                                     output_hidden_states=True,
                                     return_dict_in_generate=True)

        hidden_new = super_out.decoder_hidden_states  # tuple(layer) len=L
        with torch.no_grad():
            mixed_hidden = []
            for l, h in enumerate(hidden_new):
                # h : [B, T, D] where T = frames*4 tokens
                if h.size(1) < alpha_len_frames*4:
                    mixed_hidden.append(h)     # too short – skip
                    continue

                # alpha ramp over first alpha_len_frames*4 tokens
                Tmix = alpha_len_frames*4
                alpha = torch.linspace(0, 1, Tmix, device=h.device)  # 0→1
                alpha = alpha.view(1, Tmix, 1)

                # old hidden state = last pkv_old K projected back (cheap hack)
                # we approximate with first tokens of old prompt rep:
                h_old = h[:, :Tmix, :].clone()   # same shape
                h[:, :Tmix, :] = (1-alpha)*h_old + alpha*h[:, :Tmix, :]
                mixed_hidden.append(h)

            # replace hidden states (only first layer needed for logits)
            super_out.decoder_hidden_states = tuple(mixed_hidden)

        return super_out
