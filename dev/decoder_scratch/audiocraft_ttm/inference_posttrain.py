import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import audio_write

# 1) Load your fine-tuned checkpoint
#    This might be in an outputs/ folder created by Hydra + AudioCraft.
model_path = "outputs/2025-01-01_12-00-00/checkpoints/epoch=XX-step=XXXX.ckpt"

model = MusicGen.get_pretrained(model_path=model_path)
model = model.cuda()  # if on GPU
model.eval()

# 2) Generate music from a text prompt
prompts = ["A calm piano melody with gentle rain in the background"]
waveforms = model.generate(
    descriptions=prompts,
    progress=True,
    max_duration_s=10.0,  # seconds of audio
)

# 3) Write to disk
for i, audio in enumerate(waveforms):
    audio_write(f"text2music_{i}.wav", audio, sample_rate=32000)
