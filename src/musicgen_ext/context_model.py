# src/musicgen_ext/modeling_musicgen_ext.py
import torch
from transformers import MusicgenForConditionalGeneration

class MusicgenWithContext(MusicgenForConditionalGeneration):
    """
    Drop-in replacement that lets you
        (1) cache_past_tokens(prompt_ids)
        (2) generate_continuation(new_ids, …)
    """
    @torch.inference_mode()
    def cache_past_tokens(self, input_ids, **gen_kwargs):
        # gen_kwargs = dict(
        #     gen_kwargs,
        #     do_sample=False,
        #     max_new_tokens=0,
        #     output_hidden_states=False,
        #     output_attentions=False,
        #     use_cache=True,
        #     return_dict_in_generate=True,
        # )
        
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
    def generate_continuation(self,
                              next_input_ids=None,
                              inputs_embeds=None,
                              max_new_tokens=100,
                              streamer=None,
                              **kwargs):
        if not hasattr(self, "_pkv"):
            raise RuntimeError("Call cache_past_tokens() first.")
        return super().generate(
            next_input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            past_key_values=self._pkv,
            use_cache=True,
            **kwargs,
        )
