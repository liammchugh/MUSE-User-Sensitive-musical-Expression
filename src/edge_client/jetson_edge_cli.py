# edge_acq&encoding
import sys, os
import asyncio, json, struct, uuid, pathlib, websockets, numpy as np

# ---------- project paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.activity_encoder import ActivityEncoder
from src.utils.dataprep import ActivityDataset, AccelToRGBMel


DATA_PATH = ROOT / "data" / "PPG_ACC_processed_data" / "data_short.csv"
MODEL_DIR = ROOT / "models" / "encoder"
RESULT_DIR = ROOT / "results" / "activity_musicgen"

# ---------- activity dataset ----------
from src.utils.dataprep import ActivityDataset, AccelToRGBMel_librosa
activity_labels = {
    0:'Climbing Stairs',1:'Cycling Outdoors',2:'Driving a Car',
    3:'Lunch Break',4:'Playing Table Soccer',5:'Sitting and Reading',
    6:'Transition',7:'Walking',8:'Working at Desk'
}
ACCEL_SR = 64
SEG_S    = 8
accel_to_mel = AccelToRGBMel_librosa(sample_rate=ACCEL_SR, img_size=64, device="cpu")
dataset = ActivityDataset(DATA_PATH, "activity",
                          statics=['HeartRate','Age','Gender','Height','Weight'],
                          transform=accel_to_mel,
                          sample_rate=ACCEL_SR,
                          sample_length_s=SEG_S,
                          sliding_window_s=2)


def sample_activity():
    i = np.random.randint(len(dataset))
    return dataset[i][:2]          # img, statics

# ---------------------------------------------------------------------

SR = 32_000
PCM_SCALE = 32767

class MusicPlayer:
    def __init__(self, play=True, make_wav=True, wav_dir="results"):
        self.q_in  = asyncio.Queue()       # PCM from socket-task
        self.mix_q = asyncio.Queue()       # after cross-fade
        self.active_id = None
        self.retired_id = None
        self.prev_tail = np.zeros(0,np.float32); self.prev_id=None
        self.play = play
        self.make_wav = make_wav
        if play:
            import sounddevice as sd
            sd.default.samplerate = SR
            self.out = sd.OutputStream(channels=1, dtype="float32"); self.out.start()
        
        if self.make_wav:
            import datetime, os
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

            if self.active_id is None: # first job
                self.active_id = sid
            
            if sid == self.retired_id:
                # Skip chunks for the retired job since they're no longer needed
                print(f"[cli] skipping chunk for retired job {sid}")
                continue  # Skip processing this chunk

            if sid != self.active_id:
                # -- job change detected --
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

                # -- flush old audio --
                try:
                    while True:
                        self.mix_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            self.prev_tail = pcm[-SR // 2:]
            await self.mix_q.put(pcm)

    async def _player(self):
        while True:
            pcm = await self.mix_q.get()
            if self.play: self.out.write(pcm)
            self.wav.writeframes((pcm*PCM_SCALE)
                                 .clip(-32768,32767).astype("<i2").tobytes())
            
    def shutdown(self):
        if self.play:
            self.out.stop(); self.out.close()
        if self.make_wav:
            self.wav.close()

# ---------------------------------------------------------------------
async def main(vm_uri="ws://VM_IP:8763", mode="class", root_prompt="workout music"):
    encoder = ActivityEncoder(mode=mode, root_prompt=root_prompt)
    make_player = True
    if make_player:
        player   = MusicPlayer(play=False, wav_dir=RESULT_DIR)
    try:
        async with websockets.connect(vm_uri, max_size=None) as ws:
            async def recv():
                while True:
                    data = await ws.recv()
                    sid  = struct.unpack_from("<I", data)[0]
                    pcm  = np.frombuffer(data[4:], np.float32)
                    print(f"[cli] got {pcm.size/32000:.2f}s  from job={sid}")
                    await player.q_in.put((sid, pcm))

            if make_player:
                asyncio.create_task(recv())

            while True:
                img,stat = sample_activity()

                prompt = encoder(img, stat)[0]

                jid  = uuid.uuid4().int & 0xFFFFFFFF
                    
                safe_prompt = prompt.encode('ascii', 'ignore').decode('ascii') # get rid of unsafe characters (quotes, em dashes, etc)

                meta = json.dumps({"t":"job","id":jid,"prompt":safe_prompt})
                await ws.send(meta)                 # one frame, no tensor

                # hidden state inputs need musicgen fork edits TODO
                # prompt,hid = encoder(img, stat) 
                # jid  = uuid.uuid4().int & 0xFFFFFFFF
                # meta = json.dumps({"t":"job","id":jid,"prompt":prompt,
                #                 "dtype":"float16","shape":list(hid.shape)})
                # await ws.send(meta)
                # await ws.send(hid.numpy().tobytes())
                print(f"sent job {jid}  | prompt: {safe_prompt}")

                await asyncio.sleep(15)
        
        # graceful shutdown (only reached on normal exit)
        if make_player:
            player.wav.close()

    finally:
        # shutdown on error or Ctrl-C
        if make_player:
            player.shutdown()
    

# -- catch Ctrl-C so wave file is closed ---------------------------------
import signal
def _sigint(_, __):
    print("\n↩  Ctrl-C - closing reports and exiting")
    sys.exit(0)
signal.signal(signal.SIGINT, _sigint)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- prompt selection (blocking) ---------------------------------
    from src.utils.prompt import choose_prompt_cmd
    style_choices = [
        "techno", "pop", "rock", "soul", "jazz", "classical",
    ]
    style_prompt = choose_prompt_cmd(style_choices, default_prompt="techno")

    prompt_choices = [
        f"chill {style_prompt} music", f"workout {style_prompt} music", f"running {style_prompt} music",
        f"{style_prompt} music for focus", f"study {style_prompt} music",
        f"happy {style_prompt} music", f"sad {style_prompt} music", f"energetic {style_prompt} music"
    ]
    root_prompt = choose_prompt_cmd(prompt_choices, default_prompt="workout music")
    print(f"Selected prompt: {root_prompt}")
    import argparse, asyncio
    p=argparse.ArgumentParser(); p.add_argument("--mode",default="class")
    VM_IP = "10.206.91.197"
    print(f"VM_IP: {VM_IP} ... double check if timeout issues arise")
    p.add_argument("--vm","--uri",dest="uri",default=f"ws://{VM_IP}:8763")
    args=p.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(args.uri, args.mode, root_prompt))
    except KeyboardInterrupt:
        print("\n↩  Ctrl-C - closing WAV and exiting")
        sys.exit(0)
