# musicgen_ext/rolling_kv.py
import torch
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
        self.kv = []
        for (k, v) in past_kv:
            self.kv.append([k[:, :, -self.max_t:].contiguous(),
                            v[:, :, -self.max_t:].contiguous()])

    def replace_from(self, past_kv):
        """Replace KV with tail of provided cache (safer than append)."""
        if self.kv is None:
            return self.init_from(past_kv)
        for i, (k, v) in enumerate(past_kv):
            self.kv[i][0] = k[:, :, -self.max_t:].contiguous()
            self.kv[i][1] = v[:, :, -self.max_t:].contiguous()

    def append(self, past_kv_total):
        """Append only the delta part of a returned full-length cache."""
        if self.kv is None:
            return self.init_from(past_kv_total)

        prev_len = self.length()
        for layer, (k_new_total, v_new_total) in enumerate(past_kv_total):
            k_old, v_old = self.kv[layer]
            new_total_len = k_new_total.size(2)
            # If model returned full context, take only the tail beyond prev_len
            if new_total_len >= prev_len:
                k_delta = k_new_total[:, :, prev_len:]
                v_delta = v_new_total[:, :, prev_len:]
            else:
                # Defensive: if shorter, treat as full replacement chunk
                k_delta = k_new_total
                v_delta = v_new_total

            k_cat = torch.cat([k_old, k_delta], dim=2)
            v_cat = torch.cat([v_old, v_delta], dim=2)

            if k_cat.size(2) > self.max_t:
                k_cat = k_cat[:, :, -self.max_t:]
                v_cat = v_cat[:, :, -self.max_t:]

            self.kv[layer][0] = k_cat.contiguous()
            self.kv[layer][1] = v_cat.contiguous()

    def as_tuple(self):
        return tuple((k, v) for (k, v) in self.kv)
