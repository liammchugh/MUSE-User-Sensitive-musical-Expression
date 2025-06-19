# musicgen_live_player.py
import threading, queue, time, numpy as np, sounddevice as sd, torch
from transformers import MusicgenProcessor, BitsAndBytesConfig
from musicgen_ext.context_model import MusicgenWithContext, MusicgenWithContextAndFade
from musicgen_ext.rolling_kv   import RollingKVCache
from app import MusicgenStreamer


# ----------------------------------------------------------------------
def load_model(model_id="facebook/musicgen-small",
               bits="fp16", device="cuda"):
    if bits == "fp16":
        return MusicgenWithContext.from_pretrained(model_id).to(device).half().eval()

    if bits in {"8bit", "4bit"}:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=(bits=="8bit"),
                                     load_in_4bit=(bits=="4bit"))
        return MusicgenWithContext.from_pretrained(model_id,
                                                   quantization_config=bnb_cfg
                                                  ).to(device).eval()
    return MusicgenWithContext.from_pretrained(model_id).to(device).eval()


# ----------------------------------------------------------------------
class LiveMusicGen:
    """
    Single-instance MusicGen that can be re-prompted on-the-fly.
    """
    def __init__(self,
                 model_id="facebook/musicgen-large",
                 sys_prompt="workout techno soundtrack.",
                 bits="fp16",
                 play_steps_sec=1.0,          # size of streamer chunks
                 device=None,
                 play_audio=True,
                 wav_out=False):

        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = load_model(model_id, bits, self.device)
        self.model.__class__ = MusicgenWithContextAndFade # test path for prompt fade
        self.proc    = MusicgenProcessor.from_pretrained(model_id)

        self.sr      = self.model.audio_encoder.config.sampling_rate
        self.fr      = self.model.audio_encoder.config.frame_rate
        self.steps   = int(round(play_steps_sec * self.fr))

        # --- KV priming with static prompt ---------------------------
        sys_ids = self.proc(text=sys_prompt,
                            return_tensors="pt").input_ids.to(self.device)
        self.model.cache_past_tokens(sys_ids)
        self.kv_window = RollingKVCache(max_frames=2500)          # ≈12 s
        self.kv_window.init_from(self.model._pkv)

        # --- streamer + pcm queue -----------------------------------
        self.pcm_q  : "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._make_streamer()

        # --- audio out ----------------------------------------------
        self.play_audio = play_audio
        if play_audio:
            sd.default.samplerate = self.sr
            self.out = sd.OutputStream(channels=1, dtype="float32")
            self.out.start()

        self.wav_file = None
        if wav_out:
            import wave, pathlib
            out_dir = pathlib.Path("results/live_music"); out_dir.mkdir(parents=True, exist_ok=True)
            fname   = out_dir / f"stream_{time.strftime('%m%d_%H%M%S')}.wav"
            self.wav_file = wave.open(str(fname), "wb")
            self.wav_file.setnchannels(1); self.wav_file.setsampwidth(2); self.wav_file.setframerate(self.sr)
            print("✓ capturing WAV →", fname)

        # playback thread
        threading.Thread(target=self._playback_loop, daemon=True).start()

        # generation worker
        self._worker : threading.Thread | None = None
        self._worker_stop = threading.Event()

    # ------------------------------------------------------------------
    def _make_streamer(self):
        self.streamer = MusicgenStreamer(self.model,
                                         device=self.device,
                                         play_steps=self.steps)

        # pcm callback
        def _on_chunk(audio, *_, **__):
            try:
                self.pcm_q.put_nowait(audio.astype("float32"))
            except queue.Full:
                pass
        self.streamer.on_finalized_audio = _on_chunk

    # ------------------------------------------------------------------
    def _run_generation(self, prompt_ids):
        """
        Called inside a background thread.
        """
        try:
            out = self.model.generate_continuation(
                input_ids       = prompt_ids,
                past_key_values = self.kv_window.as_tuple(),
                streamer        = self.streamer,
                max_new_tokens  = self.steps * 15,    # ~N × play_steps_sec
                alpha_len_frames=16,                   # blend ~N*20 ms
            )
            self.kv_window.append(out.past_key_values)
        except Exception as e:
            print("gen thread exception:", e)

    # ------------------------------------------------------------------
    def push_activity(self, text_prompt: str):
        """
        Non-blocking: cancel current worker (if any) and spawn a new one.
        """
        lead_in = " Lead into the next section with no break."
        full_txt = text_prompt.strip().rstrip('.') + "." + lead_in

        ids = self.proc(text=full_txt,
                        return_tensors="pt").input_ids.to(self.device)

        # cancel old worker
        if self._worker and self._worker.is_alive():
            self._worker_stop.set()
            self._worker.join(timeout=0.2)

        # fresh streamer & stop-flag
        self._make_streamer()
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._run_generation,
                                        args=(ids,),
                                        daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def _playback_loop(self):
        try:
            while True:
                pcm = self.pcm_q.get()
                if self.play_audio and self.out.active:
                    self.out.write(pcm)
                if self.wav_file:
                    int16 = np.clip(pcm, -1, 1); int16 = (int16*32767).astype("<i2")
                    self.wav_file.writeframes(int16.tobytes())
        except Exception as e:
            print("playback exception:", e)

    def close(self):
        if self.wav_file: self.wav_file.close()
        if self.play_audio and self.out.active:
            self.out.stop(); self.out.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    sys_prompt = "Classical music for a workout session."

    player = LiveMusicGen(play_steps_sec=1.5, wav_out=True, sys_prompt=sys_prompt)

    try:
        while True:
            bpm   = int(140 + 50*np.sin(time.time()/10))
            if bpm < 120:
                label = "warm-up" 
            elif bpm < 160:
                label = "cycling"
            else:
                label = "sprint" 

            prompt = f"current activity: {label}, bpm {bpm}."
            print("→", prompt)
            full_prompt = f"{sys_prompt} ... {prompt}"
            player.push_activity(full_prompt)
            time.sleep(8)          # **change prompt every 5 s**
    except KeyboardInterrupt:
        player.close()
