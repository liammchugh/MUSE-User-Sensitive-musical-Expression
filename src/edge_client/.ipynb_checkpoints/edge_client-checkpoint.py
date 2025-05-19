import asyncio, struct, numpy as np, sounddevice as sd, json, time
from prompt_select import choose_prompt                 # ← your GUI

SR = 32_000
PCM_SCALE = 32767

async def audio_player(q, fname):
    wav = open(fname, "wb")
    import wave; ww = wave.open(wav.fileno(), "wb")
    ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(SR)

    sd.default.samplerate = SR
    out = sd.OutputStream(channels=1, dtype="float32"); out.start()

    pending = {}
    prev_id, prev_tail = None, np.zeros(0, np.float32)

    def xfade(a,b,n=int(SR*0.5)):
        if len(a)<n or len(b)<n: return np.concatenate([a,b])
        t = np.linspace(0,1,n); a[-n:]*=(1-t); b[:n]*=t; return np.concatenate([a,b])

    while True:
        sid, audio = await q.get()
        if prev_id is not None and sid != prev_id:
            audio = xfade(prev_tail, audio)
        prev_tail = audio[-SR//2:]; prev_id = sid

        out.write(audio)
        ww.writeframes((audio*PCM_SCALE).clip(-32768,32767).astype("<i2").tobytes())

async def main():
    prompt_choices = ["workout music","focus music","chill music"]
    root_prompt    = choose_prompt(prompt_choices, "workout music")

    uri = "ws://34.70.128.117:8765"
    async with websockets.connect(uri, max_size=None) as ws:
        play_q = asyncio.Queue()
        asyncio.create_task(audio_player(play_q, "capture.wav"))

        seq = 0
        while True:
            # ---- create activity packet (demo: random slice) -------------
            idx  = np.random.randint(len(dataset))
            img,stat,_ = dataset[idx]
            packet = {"t":"activity", "seq":seq,
                      "prompt":root_prompt, "statics":stat.tolist(),
                      "accel_rgb":img.numpy().tobytes()}
            await ws.send(json.dumps(packet))
            seq += 1

            # ---- pull audio ----------------------------------------------
            try:
                while True:
                    data = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    sid  = struct.unpack_from("<I", data)[0]
                    audio= np.frombuffer(data[4:], np.float32)
                    await play_q.put((sid, audio))
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
