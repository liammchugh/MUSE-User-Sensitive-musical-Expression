# musicgen_live_player.py
import threading, queue, time, numpy as np, sounddevice as sd, torch
from transformers import MusicgenProcessor, BitsAndBytesConfig
from musicgen_ext.context_model import MusicgenWithContext
from musicgen_ext.rolling_kv   import RollingKVCache
from core_prcs.app import MusicgenStreamer, RollingTokenBuffer
import inspect
from transformers.modeling_outputs import BaseModelOutput 

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
                 model_id="facebook/musicgen-small",
                 sys_prompt="workout techno soundtrack.",
                 bits="fp16",
                 play_steps_sec=1.0,          # size of streamer chunks
                 device=None,
                 play_audio=True,
                 wav_out=False):

        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = load_model(model_id, bits, self.device)
        # self.model.__class__ = MusicgenWithContextAndFade # test path for prompt fade
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
        
        self._abs_pos = self.kv_window.length()  # ADD: absolute decoder position counter
        # detect cache_position support once
        try:
            self._accepts_cache_pos = "cache_position" in inspect.signature(self.model.prepare_inputs_for_generation).parameters
        except Exception:
            self._accepts_cache_pos = False

        self.n_codebooks = self.model.decoder.num_codebooks
        self.tbuf = RollingTokenBuffer(max_frames=2500)

        # prompt encoder blending
        self._last_prompt_ids = sys_ids  # remember current text prompt ids
        with torch.no_grad():
            self._enc_last = self.model.text_encoder(
                input_ids=sys_ids,
                attention_mask=torch.ones_like(sys_ids, dtype=torch.bool, device=self.device),
                return_dict=True
            ).last_hidden_state
        self._blend = None  # no blend in progress yet

        # --- streamer + pcm queue -----------------------------------
        self.pcm_q  : "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

        # ADD (unified streamer used forever)
        self._unified_streamer = MusicgenStreamer(self.model, device=self.device, play_steps=self.steps)

        def _on_pcm(pcm, *_, stream_end=False, **kwargs):
            # still enqueue the final tail; ignore stream_end flag otherwise
            try:
                self.pcm_q.put_nowait(pcm.astype("float32"))
            except queue.Full:
                try: _ = self.pcm_q.get_nowait()
                except queue.Empty: pass
                try: self.pcm_q.put_nowait(pcm.astype("float32"))
                except queue.Full: pass

        self._unified_streamer.on_finalized_audio = _on_pcm

        # ADD prompt control & loop control
        self._next_prompt_ids = None
        self._prompt_event = threading.Event()
        self._shutdown = False
        self._prefetch_chunks = 2  # stay ~2 chunks ahead

        # Start a single long-lived generation loop
        self._gen_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self._gen_thread.start()

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

    def schedule_prompt_blend(self, new_prompt_ids: torch.LongTensor, steps: int = 3):
        with torch.no_grad():
            enc_next = self.model.text_encoder(
                input_ids=new_prompt_ids,
                attention_mask=torch.ones_like(new_prompt_ids, dtype=torch.bool, device=self.device),
                return_dict=True
            ).last_hidden_state
        # if seq lengths differ, pad/truncate the shorter one to match for a clean mix
        L = min(self._enc_last.size(1), enc_next.size(1))
        prev = self._enc_last[:, :L, :]
        next = enc_next[:, :L, :]
        self._blend = {'prev': prev, 'next': next, 'steps': max(1, int(steps)), 'i': 0}
        self._last_prompt_ids = new_prompt_ids
        print("✓ scheduled prompt blend:", new_prompt_ids)

    # ------------------------------------------------------------------
    def push_activity(self, text_prompt: str):
        ids = self.proc(text=text_prompt, return_tensors="pt").input_ids.to(self.device)

        # OPTIONAL: seed decoder if KV was reset
        seed_decoder_ids = None
        if self.kv_window.length() == 0 and self.tbuf.prompt() is not None:
            seed_decoder_ids = self.tbuf.prompt().view(1 * self.n_codebooks, -1)
        self._seed_decoder_ids = seed_decoder_ids

        # schedule a smooth blend on the encoder (if you added schedule_prompt_blend)
        if hasattr(self, "schedule_prompt_blend"):
            self.schedule_prompt_blend(ids, steps=3)
        else:
            # fallback: just remember “current prompt”
            self._last_prompt_ids = ids

        # notify the generation loop there is a new prompt
        self._next_prompt_ids = ids
        self._prompt_event.set()

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

    @torch.inference_mode()
    def _generation_loop(self):
        # ensure we have something to start from
        curr_ids = getattr(self, "_last_prompt_ids", None)
        if curr_ids is None:
            # fall back to a neutral prompt (use your sys prompt or force encoder_outputs from _enc_last)
            curr_ids = self.proc(text="ambient beat", return_tensors="pt").input_ids.to(self.device)
            self._last_prompt_ids = curr_ids

        while not self._shutdown:
            # Pick up latest prompt request, if any
            if self._prompt_event.is_set():
                ids = self._next_prompt_ids
                if ids is not None:
                    curr_ids = ids
                    self._last_prompt_ids = ids
                self._prompt_event.clear()

            # Optional: blended encoder conditioning if schedule_prompt_blend is active
            gen_kwargs = {}
            if getattr(self, "_blend", None) is not None:
                b = self._blend
                alpha = float(b['i'] + 1) / b['steps']
                enc_mix = (1.0 - alpha) * b['prev'] + alpha * b['next']      # [B, L, H]
                gen_kwargs['encoder_outputs'] = BaseModelOutput(
                    last_hidden_state=enc_mix
                )  # <- instead of (enc_mix,)
                gen_kwargs['encoder_attention_mask'] = torch.ones(
                    enc_mix.size(0), enc_mix.size(1), dtype=torch.bool, device=self.device
                )
                b['i'] += 1
                if b['i'] >= b['steps']:
                    self._enc_last = b['next']
                    self._blend = None

            # keep RoPE continuous across KV window trims
            if self._accepts_cache_pos:
                gen_kwargs['cache_position'] = torch.arange(
                    self._abs_pos, self._abs_pos + max_new_tokens, device=self.device
                )

            # reset the reusable streamer for a fresh “session”
            if hasattr(self._unified_streamer, "reset_session"):
                self._unified_streamer.reset_session()

            max_new_tokens = int(self.steps * self._prefetch_chunks)
            
            # seed decoder on cold start
            pad_id = self.model.generation_config.pad_token_id
            if pad_id is None:
                pad_id = self.model.decoder.config.pad_token_id
            if self.kv_window.length() == 0:
                gen_kwargs.setdefault(
                    "decoder_input_ids",
                    torch.full(
                        (1 * self.n_codebooks, 1),  # [B*num_codebooks, 1]
                        pad_id,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )

            try:
                out = self.model.generate_continuation(
                    input_ids=(None if 'encoder_outputs' in gen_kwargs else curr_ids),
                    past_key_values=self.kv_window.as_tuple(),
                    streamer=self._unified_streamer,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
            except Exception as e:
                print("Generation error:", type(e).__name__, str(e))
                print("  has_encoder_outputs:", 'encoder_outputs' in gen_kwargs)
                if 'encoder_outputs' in gen_kwargs:
                    eo = gen_kwargs['encoder_outputs']
                    try:
                        print("  encoder_outputs.last_hidden_state:", tuple(eo.last_hidden_state.shape))
                    except Exception:
                        print("  encoder_outputs: (no .last_hidden_state)")
                try:
                    print("  kv_len:", self.kv_window.length())
                except Exception:
                    pass
                time.sleep(0.05)
                continue

            # maintain KV & token rings for continuity
            self.kv_window.replace_from(out.past_key_values)

            if hasattr(self._unified_streamer, "token_cache") and self._unified_streamer.token_cache is not None:
                full_tokens = self._unified_streamer.token_cache.view(1, self.n_codebooks, -1)
                prev_len = 0 if self.tbuf.prompt() is None else self.tbuf.prompt().size(-1)
                self.tbuf.push(full_tokens, prev_len)
            
            # advance absolute position for the next call
            self._abs_pos += max_new_tokens

    def close(self):
        if self.wav_file: self.wav_file.close()
        if self.play_audio and self.out.active:
            self.out.stop(); self.out.close()
        self._shutdown = True
        if hasattr(self, "_gen_thread") and self._gen_thread.is_alive():
            self._gen_thread.join(timeout=1.0)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    sys_prompt = "Classical music for a workout session. Smooth transitions matching to mood of the activity."

    player = LiveMusicGen(play_steps_sec=1.5, wav_out=True, sys_prompt=sys_prompt)
    i = 0
    try:
        while True:
            bpm   = int(140 + 50*np.sin(i/2))
            if bpm < 120:
                label = "cycling warm-up" 
            elif bpm < 160:
                label = "cycling steady-state"
            else:
                label = "cycling sprint"

            prompt = f"current activity: {label}. User heart rate: {bpm}."
            print("→", prompt)
            full_prompt = f"{sys_prompt} ... {prompt}"
            player.push_activity(full_prompt)
            i += 1
            time.sleep(6)          # **change prompt every N s**
    except KeyboardInterrupt:
        player.close()
