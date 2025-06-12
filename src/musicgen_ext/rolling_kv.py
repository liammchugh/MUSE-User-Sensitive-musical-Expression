# musicgen_ext/rolling_kv.py
import torch

class RollingKVCache:
    """
    Keeps a sliding window of past_key_values for MusicGen.
    drop_stride : how many generated *frames* (50-token blocks) to remove
                  when cache grows beyond max_frames.
    """
    def __init__(self, max_frames=2500, drop_stride=200):
        self.max_t = max_frames
        self.stride = drop_stride
        self.kv = None                               # list[layers][2][B,H,T,D]

    # -----------------------------------------------------------------
    def init_from(self, past_kv):
        self.kv = [ [k.clone(), v.clone()] for k,v in past_kv ]

    # -----------------------------------------------------------------
    def append(self, new_pkv):
        if self.kv is None:                          # first call
            self.init_from(new_pkv)
            return

        for layer, (k_new, v_new) in enumerate(new_pkv):
            k_old, v_old = self.kv[layer]

            k_cat = torch.cat([k_old, k_new], dim=2)
            v_cat = torch.cat([v_old, v_new], dim=2)

            # trim if too long
            if k_cat.size(2) > self.max_t:
                k_cat = k_cat[:, :, self.stride:]
                v_cat = v_cat[:, :, self.stride:]

            self.kv[layer][0].data = k_cat
            self.kv[layer][1].data = v_cat

    # give to model.generate(...)
    def as_tuple(self):
        return tuple( (k, v) for k, v in self.kv )
