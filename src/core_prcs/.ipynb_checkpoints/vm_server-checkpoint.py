# vm_server.py  -----------------------------------------------------------
"""
GPU-side MusicGen Web-Socket server

 - listens on 0.0.0.0:8765
 - client sends   – text JSON header  ➔  binary tensor bytes
 - server streams – binary  [uint32 job_id | float32[] pcm]

One MusicGen instance is shared; only one generator can run at a time.
Older generators are cancelled when new jobs are received.
"""
import asyncio, json, struct, numpy as np, torch, websockets
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, BitsAndBytesConfig
from app import MusicgenStreamer

# ------------------------ constants -----------------------------------
SR          = 32_000
PLAY_STEPS  = int(1.0 * 50)           # ≈ N s audio chunks
MAX_STEPS   = int(15   * 50)          # 20 s total per job

# ------------------------ model ---------------------------------------
print("↻ Loading MusicGen-small on GPU …", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# musicgen = (MusicgenForConditionalGeneration
#             .from_pretrained("facebook/musicgen-small")
#             .half().to(device).eval())

bits = 'fp16' # vm currently only support half precision. Update for b&b quantization
model_id="facebook/musicgen-small"
if bits == 'fp16':
    # ---------- 16-bit model ----------
    musicgen = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device).half().eval()
elif bits == 'fp32':
    # ---------- 32-bit model ----------
    musicgen = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device).eval()
else:
    if bits == '4bit':
        # ---------- 4-bit NF4 quantized model ----------
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == '8bit':
        # ---------- 8-bit quantized model ----------
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,               # ← switched from load_in_4bit
            llm_int8_threshold=6.0,          # keep default (you can tune)
            llm_int8_enable_fp32_cpu_offload=False,  # leave on-GPU
        )
    else:
        raise ValueError(f"Invalid bit-tag: {bits}")
    musicgen = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        # device_map="auto",
    ).eval()




processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

print("✓ MusicGen ready")


# ------------------------ socket handler ------------------------------
async def handle_socket(ws: websockets.WebSocketServerProtocol):
    new_task = None
    current_task = None
    current_streamer = None

    loop = asyncio.get_running_loop()
    print(f"[srv] new WS connection from {ws.remote_address}")

    q_pcm = asyncio.Queue(maxsize=32)

    async def fanout():
        try:
            while True:
                jid, pcm = await q_pcm.get()
                payload = struct.pack("<I", jid) + pcm.astype("<f4", copy=False).tobytes()
                await ws.send(payload)
                print(f"[srv] → sent {len(pcm)/32000:.2f}s  job={jid}")
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    fanout_task = asyncio.create_task(fanout())

    try:
        while True:
            meta_raw = await ws.recv()
            meta = json.loads(meta_raw)
            print(f"[srv] meta received: {meta}")

            if meta["t"] != "job":
                continue

            job_id = int(meta["id"]) & 0xFFFFFFFF
            prompt = meta["prompt"]
            inputs = processor(text=prompt, return_tensors="pt").to("cuda")

            if current_task:
                current_task.cancel()
                if current_streamer:
                    current_streamer.stop_signal = True  # flag to end generation early
                print("Cancelled previous job")
                
                # Flush old PCM still in the queue (currently letting local player handle it)
                # while not q_pcm.empty():
                #     try:
                #         q_pcm.get_nowait()
                #     except asyncio.QueueEmpty:
                #         print("Flushed old queue")
                #         break
                        
            def thread_gen(job_id, inputs):
                nonlocal current_streamer
                nonlocal current_task
                nonlocal new_task
                streamer = MusicgenStreamer(musicgen, device="cuda", play_steps=PLAY_STEPS)
                streamer.stop_signal = False # default flag bool
                current_streamer = streamer
                
                def on_audio(pcm, *_):
                    loop.call_soon_threadsafe(asyncio.create_task, q_pcm.put((job_id, pcm.copy())))

                streamer.on_finalized_audio = on_audio
                
                
                with torch.inference_mode():
                    musicgen.generate(**inputs,
                                      streamer=streamer,
                                      max_new_tokens=MAX_STEPS)

            new_task = asyncio.create_task(asyncio.to_thread(thread_gen, job_id, inputs))
                    
                
    except Exception as e:
        print(f"[srv] exception: {e}")
    finally:
        fanout_task.cancel()
        if current_task:
            current_task.cancel()

# ------------------------ main ----------------------------------------
async def main(host="0.0.0.0", port=8765):
    print(f"WebSocket MusicGen server listening on {host}:{port}")
    async with websockets.serve(handle_socket, host, port,
                                max_size=None, ping_interval=20):
        await asyncio.Future()       # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n↩  Ctrl-C – shutting down")
