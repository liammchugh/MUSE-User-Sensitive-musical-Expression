# musicgen_ext/rolling_kv.py
import torch

class RollingKVCache:
    """
    Keeps a sliding window of past_key_values for MusicGen.
    drop_stride : how many generated *frames* (50-token blocks) to remove
                  when cache grows beyond max_frames.
    """
    def __init__(self, max_frames=2500, drop_stride=200):
        self.max_frames = max_frames
        self.stride = drop_stride
        self.kv = None                               # list[layers][2][B,H,T,D]

    # -----------------------------------------------------------------
    def init_from(self, past_kv):
        """
        Store a *deep copy* of the incoming past-key-values so we can
        roll / truncate them later.
        Each layer in MusicGen returns four tensors:
            (k_self, v_self, k_cross, v_cross)
        We keep the structure as-is.
        """
        # clone() to detach from the computation graph / GPU stream:
        self.kv = [
            [t.clone() for t in layer_kv]      # 4 tensors per layer
            for layer_kv in past_kv
        ]
        self._trim_to_window()

    def _trim_to_window(self):
        for layer in self.kv:
            # layer = [k_self, v_self, k_cross, v_cross]
            k_self, v_self = layer[0], layer[1]
            if k_self.size(-2) > self.max_frames:       # seq len dim
                k_self = k_self[..., -self.max_frames :, :]
                v_self = v_self[..., -self.max_frames :, :]
            layer[0], layer[1] = k_self, v_self

    def append(self, new_pkv):
        """
        Concatenate freshly-generated **self-attention** KV to the rolling
        window and drop the oldest frames if the window grows too long.
        """
        if self.kv is None:
            self.init_from(new_pkv)
            return

        for layer_idx, new_layer in enumerate(new_pkv):
            k_new, v_new, *_ = new_layer          # ignore cross-attn from new layer
            k_old, v_old, kx_old, vx_old = self.kv[layer_idx]

            # ---- 1.  concatenate along sequence-length dim (-2) ----------
            k_cat = torch.cat([k_old, k_new], dim=-2)
            v_cat = torch.cat([v_old, v_new], dim=-2)

            # ---- 2.  trim left if window too long ------------------------
            if k_cat.size(-2) > self.max_frames:
                trim = k_cat.size(-2) - self.max_frames
                k_cat = k_cat[..., trim:, :]
                v_cat = v_cat[..., trim:, :]

            # ---- 3.  write back, keep original cross-attn ----------------
            self.kv[layer_idx] = [k_cat, v_cat, kx_old, vx_old]

        # Optional: free the reference to `new_pkv`
        del new_pkv
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    def as_tuple(self):
        """
        Return the rolling cache in the exact tuple-of-tuples format
        expected by `model.generate(past_key_values=...)`.
        """
        return tuple(tuple(layer) for layer in self.kv)
