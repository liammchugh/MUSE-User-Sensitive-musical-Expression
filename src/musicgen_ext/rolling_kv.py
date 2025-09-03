# musicgen_ext/rolling_kv.py
import torch

try:
    from transformers.cache_utils import Cache  # HF >= 4.42
except Exception:
    Cache = None

def _to_legacy_layers(past_kv):
    """Return a tuple/list of per-layer entries, even if HF Cache is passed."""
    if Cache is not None and isinstance(past_kv, Cache):
        # New-style Cache object -> convert to legacy tuple-of-tuples
        return past_kv.to_legacy_cache()
    return past_kv  # already legacy structure

def _extract_self_kv(layer_entry):
    """
    Accepts either:
      - (k, v)
      - (k, v, cross_k, cross_v)
      - ((self_k, self_v), (cross_k, cross_v))  # nested form
    Returns (self_k, self_v).
    """
    # flat tuple/list
    if isinstance(layer_entry, (list, tuple)):
        # nested ((k,v), (k,v))
        if len(layer_entry) and isinstance(layer_entry[0], (list, tuple)):
            k, v = layer_entry[0][0], layer_entry[0][1]
            return k, v
        # flat (k, v) or (k, v, cross_k, cross_v)
        if len(layer_entry) >= 2:
            return layer_entry[0], layer_entry[1]
    raise TypeError(f"Unsupported past_kv layer format: {type(layer_entry)} / lenspec={getattr(layer_entry, '__len__', lambda: '?')()}")

class RollingKVCache:
    def __init__(self, max_frames=2500, drop_stride=200):
        self.max_t = max_frames
        self.stride = drop_stride
        self.kv = None  # list of tuples [(K, V), ...] per layer; each [B, H, T, D]

    def length(self):
        if self.kv is None:
            return 0
        return self.kv[0][0].size(2)
    
    def init_from(self, past_kv):
        layers = _to_legacy_layers(past_kv)
        self.kv = []
        for layer in layers:
            k, v = _extract_self_kv(layer)
            self.kv.append([
                k[:, :, -self.max_t:].contiguous(),
                v[:, :, -self.max_t:].contiguous()
            ])        

    def replace_from(self, past_kv):
        layers = _to_legacy_layers(past_kv)
        if self.kv is None:
            return self.init_from(layers)
        for i, layer in enumerate(layers):
            k, v = _extract_self_kv(layer)
            self.kv[i][0] = k[:, :, -self.max_t:].contiguous()
            self.kv[i][1] = v[:, :, -self.max_t:].contiguous()

    def append(self, past_kv_total):
        layers = _to_legacy_layers(past_kv_total)
        if self.kv is None:
            return self.init_from(layers)

        prev_len = self.length()
        for i, layer in enumerate(layers):
            k_new_total, v_new_total = _extract_self_kv(layer)
            k_old, v_old = self.kv[i]
            new_total_len = k_new_total.size(2)

            if new_total_len >= prev_len:
                k_delta = k_new_total[:, :, prev_len:]
                v_delta = v_new_total[:, :, prev_len:]
            else:
                # defensive fallback: replace if something got shorter
                k_delta = k_new_total
                v_delta = v_new_total

            k_cat = torch.cat([k_old, k_delta], dim=2)
            v_cat = torch.cat([v_old, v_delta], dim=2)

            if k_cat.size(2) > self.max_t:
                k_cat = k_cat[:, :, -self.max_t:]
                v_cat = v_cat[:, :, -self.max_t:]

            self.kv[i][0] = k_cat.contiguous()
            self.kv[i][1] = v_cat.contiguous()

    def as_tuple(self):
        return tuple((k, v) for (k, v) in self.kv)
