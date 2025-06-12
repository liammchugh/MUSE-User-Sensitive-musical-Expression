#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  MusicGen single-worker server with system/context prompts & rolling cache  #
# --------------------------------------------------------------------------- #

""" MusicGen server for generating music with system and context prompts.
Send updates via following structure options:
# ---- one-time system prompt ----
ws.send(json.dumps({"t":"job","id":1,"system":"calm chill-house vibe"}))

# ---- frequent context updates (no restart) ----
ws.send(json.dumps({"t":"job","id":2,"context":"transition - HR 85 bpm"}))

# ---- force a fresh song ----
ws.send(json.dumps({"t":"job","id":3,"context":"high-energy run","restart":true}))


"""


import asyncio, json, struct, argparse, torch, websockets, numpy as np
from transformers import MusicgenProcessor, BitsAndBytesConfig
from app import MusicgenStreamer

try:
    from musicgen_ext.context_model import MusicgenWithContext as MGClass
    from musicgen_ext.rolling_kv     import RollingKVCache
    HAVE_CTX = True
except ImportError:
    from transformers import MusicgenForConditionalGeneration as MGClass
    RollingKVCache, HAVE_CTX = None, False
    print("[srv] context helpers not found – KV window disabled")

SR, PLAY_STEPS, TOK_WINDOW, DROP_STRIDE = 32_000, 40, 2500, 200    # ≈0.8 s chunks

# --------------------------------------------------------------------------- #
def load_model(bits="fp16", model_id="facebook/musicgen-small"):
    if bits in {"fp16", "fp32"}:
        m = MGClass.from_pretrained(model_id, torch_dtype=torch.float16 if bits=="fp16" else None)
        return m.cuda().eval()
    cfg = BitsAndBytesConfig(load_in_8bit=True if bits=="8bit" else False,
                             load_in_4bit=True if bits=="4bit" else False,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    return MGClass.from_pretrained(model_id, quantization_config=cfg).eval()

# --------------------------------------------------------------------------- #
async def ws_handler(ws):
    loop  = asyncio.get_running_loop()
    q_pcm = asyncio.Queue(maxsize=128)

    # ---------- prompt state ----------
    system_txt  = ""           # sticky “global” prompt
    context_txt = ""           # frequently-updated prompt
    cache = RollingKVCache(TOK_WINDOW, DROP_STRIDE) if HAVE_CTX else None
    gen_task: asyncio.Task | None = None

    async def fanout():
        try:
            while True:
                jid, pcm = await q_pcm.get()
                await ws.send(struct.pack("<I", jid) + pcm.astype("<f4").tobytes())
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    fan_task = asyncio.create_task(fanout())

    try:
        while True:
            meta = json.loads(await ws.recv())
            if meta.get("t") != "job":
                continue

            jid      = int(meta["id"]) & 0xFFFFFFFF
            restart  = bool(meta.get("restart", False))
            system_txt  = meta.get("system",  system_txt)
            context_txt = meta.get("context", context_txt)
            full_prompt = (system_txt + "\n" + context_txt).strip()

            print(f"[srv] ▶ job {jid}  restart={restart}  "
                  f"(sys:{len(system_txt)} chars, ctx:{len(context_txt)} chars)")

            if restart and gen_task:
                gen_task.cancel()
                if cache: cache.clear()     # wipe past KV
                print("      KV cache cleared & generation cancelled")

            # ----------- generator thread -----------
            def run_generate():
                streamer = MusicgenStreamer(model, device=device,
                                            play_steps=PLAY_STEPS)
                streamer.on_finalized_audio = \
                    lambda pcm,*_: loop.call_soon_threadsafe(
                        asyncio.create_task, q_pcm.put((jid, pcm.copy())))

                prompt_inputs = proc(full_prompt, return_tensors="pt").to(device)
                gen_kwargs = dict(**prompt_inputs,
                                  streamer=streamer,
                                  max_new_tokens=PLAY_STEPS * 20)
                if cache and not restart and len(cache) > 0:
                    gen_kwargs["past_key_values"] = cache.as_tuple()

                with torch.inference_mode():
                    model.generate(**gen_kwargs)

                if cache is not None:
                    cache.append(model._pkv)

            gen_task = asyncio.create_task(asyncio.to_thread(run_generate))

    except websockets.ConnectionClosed:
        pass
    finally:
        fan_task.cancel()
        if gen_task: gen_task.cancel()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bits", choices=["fp16","fp32","8bit","4bit"], default="fp16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc   = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
    model  = load_model(args.bits).to(device)
    print(f"✓ MusicGen ({args.bits}) ready on {device}")

    asyncio.run(websockets.serve(ws_handler, "0.0.0.0", args.port,
                                 max_size=None, ping_interval=20))
    print(f"WebSocket server listening on :{args.port}")
    asyncio.get_event_loop().run_forever()


