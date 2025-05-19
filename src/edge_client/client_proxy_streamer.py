# client_proxy_streamer.py – dual WebSocket integration
import sys, os
import asyncio, json, struct, uuid, pathlib, websockets, numpy as np, sounddevice as sd

# ---------- project paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.activity_encoder import ActivityEncoder
from src.utils.dataprep import ActivityDataset, AccelToRGBMel

DATA_PATH = ROOT / "data" / "PPG_ACC_processed_data" / "data.pkl"
MODEL_DIR = ROOT / "models" / "encoder"
RESULT_DIR = ROOT / "results" / "streaming_musicgen"

SR = 32_000
PCM_SCALE = 32767

class MusicPlayer:
    def __init__(self, play=True, wav_dir="results"):
        self.q_in  = asyncio.Queue()
        self.mix_q = asyncio.Queue()
        self.active_id = None
        self.retired_id = None
        self.prev_tail = np.zeros(0, np.float32)
        self.play = play
        if play:
            sd.default.samplerate = SR
            self.out = sd.OutputStream(channels=1, dtype="float32"); self.out.start()

        import datetime
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        wave_file = f"stream_{now}.wav"
        import wave
        os.makedirs(wav_dir, exist_ok=True)
        self.wav = wave.open(os.path.join(wav_dir, wave_file), "wb")
        self.wav.setnchannels(1); self.wav.setsampwidth(2); self.wav.setframerate(SR)

        asyncio.create_task(self._mixer())
        asyncio.create_task(self._player())

    async def _mixer(self):
        while True:
            sid, pcm = await self.q_in.get()

            if self.active_id is None:
                self.active_id = sid

            if sid == self.retired_id:
                print(f"[cli] skipping chunk for retired job {sid}")
                continue

            if sid != self.active_id:
                print(f"[cli] switching to job {sid} from {self.active_id}")
                chunk_len = len(pcm)
                if len(pcm) >= chunk_len and len(self.prev_tail) >= chunk_len:
                    t = np.linspace(0, 1, chunk_len, dtype=np.float32)
                    pcm[:chunk_len] *= t
                    self.prev_tail[-chunk_len:] *= 1 - t
                    pcm = np.concatenate([self.prev_tail, pcm])
                else:
                    print(f"[cli] skipping crossfade: short segment")
                self.prev_tail = np.zeros(0, np.float32)
                self.retired_id = self.active_id
                self.active_id = sid
                try:
                    while True: self.mix_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            self.prev_tail = pcm[-SR // 2:]
            await self.mix_q.put(pcm)

    async def _player(self):
        while True:
            pcm = await self.mix_q.get()
            if self.play: self.out.write(pcm)
            self.wav.writeframes((pcm * PCM_SCALE).clip(-32768, 32767).astype("<i2").tobytes())

    def shutdown(self):
        if self.play:
            self.out.stop(); self.out.close()
        self.wav.close()

# ---------------------------------------------------------------------
async def main(proxy_recv_uri="ws://localhost:8763", vm_uri="ws://VM_IP:8765"):
    player = MusicPlayer(play=True, wav_dir=RESULT_DIR)

    async with websockets.connect(proxy_recv_uri, max_size=None) as proxy_ws, \
               websockets.connect(vm_uri, max_size=None) as vm_ws:

        async def recv_audio():
            while True:
                data = await vm_ws.recv()
                sid = struct.unpack_from("<I", data)[0]
                pcm = np.frombuffer(data[4:], np.float32)
                print(f"[cli] got {pcm.size/32000:.2f}s from job={sid}")
                await player.q_in.put((sid, pcm))

        async def relay_prompts():
            while True:
                prompt_json = await proxy_ws.recv()
                try:
                    meta = json.loads(prompt_json)
                    print(f"[cli] → forwarding: {meta['id']} | {meta['prompt'][:50]}...")
                    await vm_ws.send(prompt_json)
                except Exception as e:
                    print("[cli] failed to forward prompt:", e)

        await asyncio.gather(recv_audio(), relay_prompts())

# ---------------------------------------------------------------------
import signal, sys

def _sigint(_, __):
    print("\n↩  Ctrl-C - shutting down")
    sys.exit(0)
signal.signal(signal.SIGINT, _sigint)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="ws://localhost:8763")
    parser.add_argument("--vm", default="ws://35.238.209.234:8765")
    args = parser.parse_args()
    asyncio.run(main(proxy_recv_uri=args.proxy, vm_uri=args.vm))
