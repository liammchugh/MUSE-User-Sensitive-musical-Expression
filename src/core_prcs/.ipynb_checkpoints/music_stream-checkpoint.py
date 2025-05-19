import queue, threading, time, math, numpy as np, sounddevice as sd, torch
from queue import Queue
from threading import Thread
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from transformers import BitsAndBytesConfig
from app import MusicgenStreamer          # your streamer class

CHUNK_SEC  = 1.5         # ≈2 × latency of first sound
CROSS_MS = CHUNK_SEC / 1.5      # cross-fade window (s)
MAX_ACTIVE = 1         # Keep only newest generator running (only enacted after new gen's first chunk)
GLOBAL_QUEUE = 32      # max size of global queue
MAX_TIME = 16          # max time to generate from single model (s)
PCM_SCALE = 32767      # max of int16


def crossfade(a, b, fs, ms=CROSS_MS):
    n = int(fs * ms)
    if n == 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a[-n:] *= 1 - fade
    b[:n]  *= fade
    return np.concatenate([a, b])

class GeneratorHandle:
    """Encapsulate one MusicGen run + its local queue."""
    def __init__(self, model, processor, prompt, play_steps, max_steps, device):
        self.queue     = Queue(maxsize=int(GLOBAL_QUEUE))
        self.stop_evt  = threading.Event()

        # build streamer
        streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)
        streamer_id = id(self)
        def _cb(audio, stream_end=False, stop=self.stop_evt, tag=streamer_id):
            if stop.is_set(): return
            try:
                self.queue.put_nowait((tag, audio.astype("float32")))
            except queue.Full: pass
        streamer.on_finalized_audio = _cb

        # launch generation thread
        inputs = processor(text=prompt, return_tensors="pt").to(device)

        def _generate():
            with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
                model.generate(**inputs, streamer=streamer,
                            max_new_tokens=max_steps,)
        Thread(target=_generate, daemon=True).start()

    def cancel(self):
        self.stop_evt.set()

    def exhausted(self):
        return self.stop_evt.is_set() and self.queue.empty()
    

class MUSE_Activity_Player:
    def __init__(self, model_id="facebook/musicgen-small", bits='fp16', device=None, save_audio=True, play_out=False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if bits == 'fp16':
            # ---------- 16-bit model ----------
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device).half().eval()
        elif bits == 'fp32':
            # ---------- 32-bit model ----------
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device).eval()
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
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                # device_map="auto",
            ).eval()


        self.processor = MusicgenProcessor.from_pretrained(model_id)
        self.device    = device
        self.save_audio = save_audio
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate
        self.need_reset = False # reset queue after successful prompt switch

        # queues
        self.mix_q   = Queue(maxsize=GLOBAL_QUEUE)     # global queue for playback
        self.handles = []                    # active GeneratorHandle objects
        self.lock    = threading.Lock()
        self.h_newflag = False

        # audio output
        sd.default.samplerate = self.sampling_rate
        self.play_out = play_out
        if self.play_out:
            self.out = sd.OutputStream(channels=1, dtype="float32", samplerate=self.sampling_rate)
            self.out.start()

        # audio capture
        if save_audio:
            import wave, pathlib
            import atexit
            script_dir = pathlib.Path(__file__).resolve().parent
            project_root = script_dir.parent.parent # Go up two levels from src/edge_prcss
            import datetime
            now = datetime.datetime.now()
            formatted_date = now.strftime("%m-%d_%H-%M")
            self.wav_path = project_root / "results" / "gen_music" / f"stream_capture_{formatted_date}.wav"
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)

            self.wav_file = wave.open(str(self.wav_path), "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2)          # int16 → 2 bytes
            self.wav_file.setframerate(self.sampling_rate)
            # atexit.register(self._close_wav)
            print("✓ WAV file will be saved at", self.wav_path)


        # start mixer + playback threads
        Thread(target=self._mixer_loop,    daemon=True).start()
        Thread(target=self._playback_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def push_activity(self, prompt, chunk_sec=CHUNK_SEC):
        play_steps = int(chunk_sec * self.frame_rate)
        max_steps  = int(MAX_TIME * self.frame_rate)
        h_new = GeneratorHandle(self.model, self.processor, prompt,
                                play_steps, max_steps, self.device)
        
        with self.lock:
            self.handles.append(h_new)

        # spin-wait for first chunk of new model
        while h_new.queue.qsize() == 0:
            time.sleep(0.005)
            if h_new.queue.qsize() > 0:
                self.h_newflag = True

        with self.lock:
            old = self.handles[0]
            # old.queue = Queue()       # drop the queue to stop further reads
            while len(self.handles) > MAX_ACTIVE:
                self.need_reset = True 
                old.cancel()                   # stop further callbacks
                self.handles.pop(0)            # drop from the active list
                print(f"Popped and cleared handle {id(old)}")

    # ------------------------------------------------------------------
    def _mixer_loop(self):
        """Pull chunks from active generators, apply cross-fade, push to mix_q."""
        prev_chunk = np.zeros(0, dtype=np.float32)
        prev_tag   = None

        while True:
            with self.lock:
                # drop finished handles
                self.handles = [h for h in self.handles if not h.exhausted()]
                active = list(self.handles)
            if not active:
                time.sleep(0.005)
                continue

            # read from first generator that has data
            got_tag, got_samples = None, None
            for h in active:
                try:
                    got_tag, got_samples = h.queue.get_nowait()
                except queue.Empty:
                    continue
            if got_tag is None:
                time.sleep(0.005)
                continue

            # new prompt-worker chunk: cross-fade with previous tail
            if prev_tag is not None and got_tag != prev_tag:
                mixed = crossfade(prev_chunk, got_samples, self.sampling_rate)
                # hard reset after a prompt switch
                # if getattr(self, "need_reset", True):
                    # prev_chunk = np.zeros(0, dtype=np.float32)
                # prev_tag   = None
                # self.need_reset = False
                with self.mix_q.mutex:
                    self.mix_q.queue.clear()   # drop anything already queued
                    print("Resetting mix queue.")
                while len(self.handles) > MAX_ACTIVE:
                    self.need_reset = True 
                    self.handles[0].cancel()       # stop further callbacks
                    self.handles.pop(0)            # drop from the active list
                    print(f"Popped and cleared handle {id(self.handles[0])} from mixerloop")

            else: # same prompt → just append
                # mixed = np.concatenate([prev_chunk, got_samples])
                mixed = got_samples
                
            prev_chunk = mixed[-int(self.sampling_rate*CROSS_MS):]  # save tail
            prev_tag = got_tag

            self.mix_q.put(mixed, block=True)

    # ------------------------------------------------------------------
    def _playback_loop(self):
        try:
            while True:
                chunk = self.mix_q.get()
                if self.play_out:
                    self.out.write(chunk)
                if self.save_audio:
                    int16 = np.clip(chunk, -1.0, 1.0)          # keep in range
                    int16 = (int16 * PCM_SCALE).astype("<i2")  # little-endian int16
                    self.wav_file.writeframes(int16.tobytes())

                print(f"queue: {self.mix_q.qsize():2}/{self.mix_q.maxsize} | "
                f"handles: {len(list(self.handles))} | Time: {time.strftime('%H:%M:%S')}")
            
        except KeyboardInterrupt:
            print("Playback interrupted by user.")
        finally:
            print("Playback loop exited.")

    def shutdown(self):
        # 1) stop all generators
        with self.lock:
            for h in self.handles:
                h.cancel()

        # 2) stop audio stream
        if self.play_out:
            if self.out.active:
                self.out.stop()
            self.out.close()

        # 3) close the capture file
        if self.save_audio:
            self.wav_file.close()
            print("✓ WAV file saved at", self.wav_path)

def _graceful_exit(sig, frame):
    print("\n↩  Caught Ctrl-C — shutting down …")
    player.shutdown()
    sys.exit(0)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import signal, sys

    signal.signal(signal.SIGINT,  _graceful_exit)   # Ctrl-C / kill -2
    signal.signal(signal.SIGTERM, _graceful_exit)   # kill
        

    player = MUSE_Activity_Player()

    import logging
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

    def heartbeat():
        base = "Cycling workout - music excitement level matching heartrate, with range 60-180."
        hr_last = 60
        while True:
            hr = int((math.sin(time.time()/(4*math.pi)) + 1) * 60 + 60)
            condition = 'Previous heartrate was {hr_last} BPM'
            prompt = f"{base} ... {condition} ... Current Heartrate {hr} BPM."
            print(f"Pushing: {prompt}")
            player.push_activity(prompt)
            hr_last = hr
            time.sleep(15)

    threading.Thread(target=heartbeat, daemon=True).start()
    while True:
        time.sleep(1)
