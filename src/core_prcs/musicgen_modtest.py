from musicgen_ext.rolling_kv import RollingKVCache
from musicgen_ext.context_model import MusicgenWithContext   # from previous answer
proc   = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
model  = MusicgenWithContext.from_pretrained("facebook/musicgen-small").half().cuda().eval()

# ---------- static system prompt -----------------------------------
sys_ids = proc("workout techno soundtrack.", return_tensors="pt").input_ids.to(model.device)
model.cache_past_tokens(sys_ids)             # fills model._pkv

kv_window = RollingKVCache(max_frames=2500)  # ~12 s history
kv_window.init_from(model._pkv)              # prime with static prompt

streamer = MusicgenStreamer(model, device="cuda", play_steps=40)

while True:
    # ---- every activity event -------------------------------------
    dyn_txt = f"current activity: {activity_label}, bpm {bpm}."
    dyn_ids = proc(dyn_txt, return_tensors="pt").input_ids.to(model.device)

    out = model.generate_continuation(
        input_ids=dyn_ids,
        past_key_values=kv_window.as_tuple(),
        streamer=streamer,
        max_new_tokens=50*play_seconds
    )

    kv_window.append(out.past_key_values)     # roll the window
