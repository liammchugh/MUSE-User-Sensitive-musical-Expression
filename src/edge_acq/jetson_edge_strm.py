# jetson_edge_strm.py — Jetson client → Client (laptop/phone) proxy WebSocket
import sys, os
import asyncio, json, struct, uuid, pathlib, websockets, numpy as np

# ---------- project paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.activity_encoder import ActivityEncoder
from src.utils.dataprep import ActivityDataset, AccelToRGBMel_librosa

DATA_PATH = ROOT / "data" / "PPG_ACC_processed_data" / "data_short_diverse_activity.csv"
MODEL_DIR = ROOT / "models" / "encoder"
RESULT_DIR = ROOT / "results" / "activity_musicgen"

activity_labels = {
    0: 'Climbing Stairs', 1: 'Cycling Outdoors', 2: 'Driving a Car',
    3: 'Lunch Break', 4: 'Playing Table Soccer', 5: 'Sitting and Reading',
    6: 'Transition', 7: 'Walking', 8: 'Working at Desk'
}
ACCEL_SR = 64
SEG_S = 8
accel_to_mel = AccelToRGBMel_librosa(sample_rate=ACCEL_SR, img_size=64, device="cpu")
dataset = ActivityDataset(DATA_PATH, "activity",
                          statics=['HeartRate', 'Age', 'Gender', 'Height', 'Weight'],
                          transform=accel_to_mel,
                          sample_rate=ACCEL_SR,
                          sample_length_s=SEG_S,
                          sliding_window_s=2)

def sample_activity():
    i = np.random.randint(len(dataset))
    return dataset[i]  # img, statics, label

def visualize_sample(img_tensor, pred_label, true_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)

    # Normalize each channel to [0,1]
    for c in range(img.shape[2]):
        channel = img[..., c]
        min_val, max_val = np.min(channel), np.max(channel)
        if max_val > min_val:
            img[..., c] = (channel - min_val) / (max_val - min_val)
        else:
            img[..., c] = 0.0  # flat channel

    save_dir = ROOT / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"spectrogram.png"

    plt.figure(figsize=(4, 4))
    plt.imshow(img, extent=[0, SEG_S, 0, ACCEL_SR // 2], origin='lower', aspect='auto')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

#     plt.axis('off')
    plt.title(f"Predicted: {pred_label}\nGround Truth: {true_label}")
    plt.savefig(str(save_path), bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------
async def main(proxy_uri="ws://CLI_IP:8763", mode="class", root_prompt="workout music"):
    encoder = ActivityEncoder(mode=mode, root_prompt=root_prompt)
    print(f"Built encoder in {mode} mode.")

    try:
        async with websockets.connect(proxy_uri, max_size=None) as ws:
            img, stat, label = sample_activity()
            prompt, pred_label = encoder(img, stat)
            print(f"LABEL PREDICTION {pred_label}")
            visualize_sample(img, pred_label, activity_labels[label.item()])

            while True:
                img, stat, label = sample_activity()
                prompt, pred_label = encoder(img, stat)

                jid = uuid.uuid4().int & 0xFFFFFFFF
                safe_prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                meta = json.dumps({"t": "job", "id": jid, "prompt": safe_prompt})
                await ws.send(meta)
                print(f"[jetson] sent job {jid} | prompt: {safe_prompt}")
                await asyncio.sleep(20)

    except Exception as e:
        print(f"[jetson] connection error: {e}")

# ---------------------------------------------------------------------
import signal

def _sigint(_, __):
    print("\n↩  Ctrl-C - exiting")
    sys.exit(0)
signal.signal(signal.SIGINT, _sigint)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils.prompt import choose_prompt_cmd
    style_choices = ["techno", "pop", "rock", "soul", "jazz", "classical"]
    style_prompt = choose_prompt_cmd(style_choices, default_prompt="techno")

    prompt_choices = [
        f"chill {style_prompt} music", f"workout {style_prompt} music", f"running {style_prompt} music",
        f"{style_prompt} music for focus", f"study {style_prompt} music",
        f"happy {style_prompt} music", f"sad {style_prompt} music", f"energetic {style_prompt} music"
    ]
    root_prompt = choose_prompt_cmd(prompt_choices, default_prompt="workout music")
    print(f"Selected prompt: {root_prompt}")

    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--mode", default="class")
    CLI_IP = "10.206.91.197"  # replace with your CLIENT IP
    print(f"CLI_IP: {CLI_IP} ... check if timeout issues arise")
    p.add_argument("--proxy", dest="uri", default=f"ws://{CLI_IP}:8763")
    args = p.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(proxy_uri=args.uri, mode=args.mode, root_prompt=root_prompt))
    except KeyboardInterrupt:
        print("\n↩  Ctrl-C - exiting")
        sys.exit(0)
