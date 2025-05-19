"""
Generate and *stream* MusicGen audio conditioned on live activity
samples.

Run one of

    python -m edge_prcss.activity_stream --mode class   # text-label prompt
    python -m edge_prcss.activity_stream --mode embed   # SigLIP embedding

The script listens to “activity events” that you push into the player
(`player.push_activity(sample)`) and keeps exactly one active generator
at a time (cross-fading when a new sample arrives).
"""
# ----------------------------------------------------------------------
import argparse, datetime, math, random, sys, time, pathlib, queue, threading
from   queue      import Queue
from   threading  import Thread

import numpy  as np
import torch
from   transformers import (MusicgenForConditionalGeneration,
                            MusicgenProcessor, BitsAndBytesConfig)
from app import MusicgenStreamer          # streamer class


# ---------- project paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_PATH = ROOT / "data" / "PPG_ACC_processed_data" / "data.pkl"
MODEL_DIR = ROOT / "models" / "encoder"
RESULT_DIR = ROOT / "results" / "activity_musicgen"

# ---------- activity dataset ----------
from src.utils.dataprep import ActivityDataset, AccelToRGBMel
activity_labels = {
    0:'Climbing Stairs',1:'Cycling Outdoors',2:'Driving a Car',
    3:'Lunch Break',4:'Playing Table Soccer',5:'Sitting and Reading',
    6:'Transition',7:'Walking',8:'Working at Desk'
}
ACCEL_SR = 64
SEG_S    = 8
accel_to_mel = AccelToRGBMel(sample_rate=ACCEL_SR, img_size=64, device="cpu")
dataset = ActivityDataset(DATA_PATH, "ActivityDescr",
                          statics=['HeartRate','Age','Gender','Height','Weight'],
                          transform=accel_to_mel,
                          sample_rate=ACCEL_SR,
                          sample_length_s=SEG_S,
                          sliding_window_s=2)


# ========== prompt builders ===========================================
def build_prompt_from_classifier(root_prompt, img, statics, classifier, return_label=False):
    heart_rate = statics[0].item()    # Get the heart rate from statics
    
    with torch.inference_mode():
        pred = classifier(img.unsqueeze(0), statics.unsqueeze(0))
        label = activity_labels[pred.argmax(-1).item()]
    
    prompt = (f"{root_prompt}… Current activity: {label.lower()}."
               f" Heart rate: {heart_rate:.0f} bpm.")
    
    return (prompt, label) if return_label else prompt

def build_encoder_hid_from_siglip(img,
                                  siglip_proc, siglip_enc,
                                  text_tok, text_enc, device):
    #  Vision → 768
    v = siglip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        v_emb = siglip_enc.get_image_features(**v)           # (1,768)
    v_emb = v_emb.to(device).half()

    #  Text → hidden states (1,T,768)
    t_h = text_enc(**text_tok.to(device)).last_hidden_state
    return torch.cat([t_h, v_emb[:,None,:]], dim=1)          # (1,T+1,768)


# ========== streamer player ===========================================
CHUNK_SEC  = 2.0
CROSS_MS   = int(CHUNK_SEC * 1e3 / 1.5)
GLOBAL_Q   = 32
MAX_TIME_S = 16
PCM_SCALE  = 32767

def crossfade(a,b,fs, ms=CROSS_MS):
    n = int(fs*ms/1e3)
    if n==0 or len(a)<n or len(b)<n: return np.concatenate([a,b])
    f = np.linspace(0,1,n, dtype=np.float32)
    a[-n:] *= 1-f;  b[:n] *= f
    return np.concatenate([a,b])

class GeneratorHandle:
    def __init__(self, musicgen, streamer, kwargs):
        self.q        = Queue(maxsize=GLOBAL_Q//2)
        self.stop_evt = threading.Event()

        # attach callback
        tag = id(self)
        def _cb(audio, stream_end=False, stop=self.stop_evt):
            if stop.is_set(): return
            try: self.q.put_nowait((tag, audio.astype("float32")))
            except queue.Full: pass
        streamer.on_finalized_audio = _cb

        def _worker():
            with torch.inference_mode():
                musicgen.generate(streamer=streamer, **kwargs)
        Thread(target=_worker, daemon=True).start()

    def cancel(self):     self.stop_evt.set()
    def exhausted(self):  return self.stop_evt.is_set() and self.q.empty()

class ActivityMusicPlayer:
    def __init__(self, root_prompt="workout music", mode="class", bits="fp16", device=None,
                 save_result=True, play_audio=False):
        self.root_prompt = root_prompt
        self.mode   = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if play_audio:
            import sounddevice as sd
        # ---- MusicGen --------------------------------------------------
        if bits=="fp16":  quant_cfg = None
        elif bits=="fp32": quant_cfg = None
        elif bits=="8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("bits must be fp16|fp32|8bit")

        self.musicgen  = MusicgenForConditionalGeneration.from_pretrained(
                            "facebook/musicgen-small",
                            quantization_config=quant_cfg).to(self.device).eval()
        self.mproc     = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
        self.sr        = self.musicgen.audio_encoder.config.sampling_rate
        self.frame_r   = self.musicgen.audio_encoder.config.frame_rate
        # ---- prompt helpers -------------------------------------------
        if mode=="class":
            from models.encoder.encoder import SimpleCNN
            ckpt = next(MODEL_DIR.glob("cnn_classifier_*.pth"))
            sample_statics = dataset[0][1]
            self.classifier = SimpleCNN(len(activity_labels),
                                        sample_statics.shape[0], img_size=64)
            self.classifier.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.classifier.eval()
        else:  # siglip
            from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModel as HFModel
            self.siglip_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.siglip_enc  = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
            ckpt = MODEL_DIR / "SigLIP_seglen8.pth"
            self.siglip_enc.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            self.t5_tok     = AutoTokenizer.from_pretrained("t5-base")
            self.text_tok   = self.t5_tok(f"{self.root_prompt}… ", return_tensors="pt")
            self.text_enc   = self.musicgen.text_encoder

        # ---- streaming infra ------------------------------------------
        self.mix_q   = Queue(maxsize=GLOBAL_Q)
        self.handles = []
        self.lock    = threading.Lock()

        self.play_audio = play_audio
        if play_audio:
            sd.default.samplerate = self.sr
            self.out = sd.OutputStream(channels=1, dtype="float32", samplerate=self.sr)
            self.out.start()

        self.save_result = save_result
        if self.save_result:
            import wave

            ts = datetime.datetime.now().strftime("%m%d_%H%M")
            path = RESULT_DIR / f"stream_{ts}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.wav = wave.open(str(path),"wb");  self.wav.setnchannels(1)
            self.wav.setsampwidth(2);             self.wav.setframerate(self.sr)
            print("✓ saving capture to", path)

            # ---- run-log ------------------------------------------------
            self.report_path  = path.with_suffix(".txt")
            self.report_file  = open(self.report_path, "a", encoding="utf-8")
            self.report_file.write(
                f"# Activity-to-Music session started {ts}\n"
                f"# audiogen sample_rate={self.sr}\n"
                "#  offset(s) | label | prompt\n")
        Thread(target=self._mixer , daemon=True).start()
        Thread(target=self._player, daemon=True).start()

    # -------- push one activity sample --------------------------------
    def push_activity(self, root_prompt, img, statics):
        """spawn a MusicGen generator for this sample and fade over"""
        if root_prompt and root_prompt != self.root_prompt:
            self.root_prompt = root_prompt
            print("✓ root_prompt changed to", root_prompt)
            if self.save_result:
                self.report_file.write(f"# root_prompt changed to {root_prompt}\n")
                self.report_file.flush()
            if self.mode == "embed": 
                self.text_tok = self.t5_tok(f"{self.root_prompt}… ", return_tensors="pt")
        
        root_prompt = root_prompt or self.root_prompt
        if self.mode=="class":
            prompt, label = build_prompt_from_classifier(self.root_prompt,
                                img, statics, self.classifier, return_label=True)
            inputs = self.mproc(text=prompt, return_tensors="pt").to(self.device)
            kwargs = dict(**inputs, max_new_tokens=int(self.frame_r*MAX_TIME_S))
            streamer = MusicgenStreamer(self.musicgen, device=self.device,
                                        play_steps=int(CHUNK_SEC*self.frame_r))
        else:  # embed
            enc_hid = build_encoder_hid_from_siglip(
                        img, self.siglip_proc, self.siglip_enc,
                        self.text_tok, self.text_enc, self.device)
            kwargs   = dict(encoder_hidden_states=enc_hid,
                            max_new_tokens=int(self.frame_r*MAX_TIME_S))
            streamer = MusicgenStreamer(self.musicgen, device=self.device,
                                        play_steps=int(CHUNK_SEC*self.frame_r))
            label = "EMBED"

        if self.save_result:
            offset = self.wav.getnframes() / self.sr           # seconds at start
            self.report_file.write(f"{offset:9.2f} | {label:<22} | {prompt}\n")
            self.report_file.flush()

        h = GeneratorHandle(self.musicgen, streamer, kwargs)
        with self.lock:
            self.handles.append(h)

        # wait for first chunk then drop old handle
        while h.q.qsize()==0: time.sleep(0.005)
        with self.lock:
            while len(self.handles)>1:
                old=self.handles.pop(0);  old.cancel()
                    # ── NEW: drain leftovers from the old handle ---------------------
                while not old.q.empty():
                    try: old.q.get_nowait()
                    except queue.Empty: break

                # ── NEW: flush anything it had already pushed to the global queue
                with self.mix_q.mutex:
                    self.mix_q.queue.clear()

    # -------- mixer / player threads ----------------------------------
    def _mixer(self):
        prev, prev_id = np.zeros(0,dtype=np.float32), None
        while True:
            with self.lock:
                self.handles=[h for h in self.handles if not h.exhausted()]
                act=list(self.handles)
            if not act: time.sleep(0.005); continue
            tag,x=None,None
            for h in act:
                try: tag,x=h.q.get_nowait(); break
                except queue.Empty: pass
            if tag is None: time.sleep(0.01); continue
            
            if prev_id and tag!=prev_id:
                out = crossfade(prev,x,self.sr)
                prev = np.zeros(0,dtype=np.float32)
                prev_id = tag
                with self.mix_q.mutex:
                    self.mix_q.queue.clear()
            else: 
                out = x
                prev, prev_id = out[-int(self.sr*CROSS_MS/1e3):], tag
            self.mix_q.put(out)

    def _player(self):
        try:
            while True:
                chunk=self.mix_q.get()
                if self.play_audio: self.out.write(chunk)
                if self.save_result:
                    pcm=(np.clip(chunk,-1,1)*PCM_SCALE).astype("<i2")
                    self.wav.writeframes(pcm.tobytes())
                
                print(f"queue: {self.mix_q.qsize():2}/{self.mix_q.maxsize} | "
                f"handles: {len(list(self.handles))} | Time: {time.strftime('%H:%M:%S')}")

        except KeyboardInterrupt: pass

    def shutdown(self):
         if self.save_result:
            self.wav.close()
            self.report_file.close()
            print("✓ WAV file saved at", self.wav._file.name)

# ----------------------------------------------------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["class","embed"],default="class")
    parser.add_argument("--bits",choices=["fp16","fp32","8bit"],default="fp16")
    args=parser.parse_args()

    # --- prompt selection (blocking) ---------------------------------
    from prompt_select import choose_prompt
    # prompt_choices = [
    #     "chill music", "workout music", "running music",
    #     "focus music",  "study music",   "happy music",
    #     "sad music",    "energetic music", "calm music"
    # ]
    # root_prompt = choose_prompt(prompt_choices, default_prompt="workout music")
    root_prompt = "chill music"
    print("✓ starting session with root_prompt =", root_prompt)

    player = ActivityMusicPlayer(mode=args.mode, bits=args.bits,
                                 save_result=True, play_audio=False)

    # demo: push a random sample every 15 s -----------------------------
    def heartbeat():
        while True:
            idx=random.randrange(len(dataset))
            img,stat,_=dataset[idx]
            player.push_activity(root_prompt, img, stat)
            time.sleep(15)
    
    Thread(target=heartbeat,daemon=True).start()

    while True: time.sleep(1)
