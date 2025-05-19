#!/usr/bin/env python

import sys
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import audio_write

def generate_music_from_text(
    prompts,
    model_version: str = "small",
    duration: float = 10.0,
    output_prefix: str = "musicgen_output"
):
    """
    Generate music from a list of text prompts using MusicGen's pretrained model.

    Args:
        prompts (List[str]): Text descriptions for the music you want to generate.
        model_version (str): Pretrained model version: ["small", "medium", "large", "melody"] 
                             or path to your fine-tuned checkpoint.
        duration (float): Duration of generated audio in seconds.
        output_prefix (str): Prefix for the output .wav files.

    Returns:
        List[torch.Tensor]: List of waveforms generated (1D or 2D if stereo).
    """
    print(f"Loading pretrained MusicGen model: {model_version}")
    model = MusicGen.get_pretrained(model_version)
    model.set_generation_params(duration=duration)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    print(f"Generating music for prompts: {prompts}")
    with torch.no_grad():
        waveforms = model.generate(descriptions=prompts)

    sample_rate = model.sample_rate  # Typically 32000 for MusicGen
    for i, (audio, prompt) in enumerate(zip(waveforms, prompts)):
        out_file = f"{output_prefix}_{i}.wav"
        print(f"  -> Saving prompt '{prompt}' to {out_file}")
        audio_write(out_file, audio, sample_rate=sample_rate)

    return waveforms

if __name__ == "__main__":
    """
    Usage:
      python inference_pretrain.py "A calm piano melody" 15

    1. If the user supplies a text prompt and an optional duration from the CLI, use them.
    2. Otherwise, fallback to default values.
    """
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "An uplifting classical track with prominent, grand swells."

    if len(sys.argv) > 2:
        duration = float(sys.argv[2])
    else:
        duration = 10.0

    # Generate from a single prompt; you can pass multiple prompts as a list.
    generate_music_from_text([prompt], model_version="small", duration=duration)
