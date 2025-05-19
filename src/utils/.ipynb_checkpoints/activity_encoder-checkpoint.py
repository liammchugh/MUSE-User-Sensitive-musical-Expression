import pathlib, torch
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

MODEL_DIR = ROOT / "models" / "encoder"
activity_labels = {0:'Climbing Stairs',1:'Cycling Outdoors',2:'Driving a Car',
                   3:'Lunch Break',4:'Playing Table Soccer',5:'Sitting and Reading',
                   6:'Transition',7:'Walking',8:'Working at Desk'}


def build_prompt_from_classifier(root_prompt, img, statics, classifier, activity_labels, return_label=True):
    heart_rate = statics[0].item()    # Get the heart rate from statics
    age = statics[1].item()
    gender = statics[2].item()
    height = statics[3].item()
    weight = statics[4].item()
    
    with torch.inference_mode():
        pred = classifier(img.unsqueeze(0), statics.unsqueeze(0))
        print(f"Predicted Logits: {pred}")
        label = activity_labels[pred.argmax(-1).item()]
#     with torch.inference_mode():
#         logits = classifier(img.unsqueeze(0), statics.unsqueeze(0)).squeeze(0)
#         # Sort indices descending by logit
#         sorted_indices = torch.argsort(logits, descending=True)
#         for idx in sorted_indices:
#             if idx.item() != 6:  # Skip "Transition" (label 6)
#                 label = activity_labels[idx.item()]
#                 break
#         else:
#             label = activity_labels[6]  # fallback (should rarely hit)

    prompt = (
        f"Music in the style of {root_prompt}. "
        f"The listener is {age:.0f} years old, {height:.0f} cm tall, and weighs {weight:.1f} kg. "
        f"Heart rate is {heart_rate:.0f} bpm. "
        f"Compose music that reflects the listeners style and physiological state. "
        f"Current activity: {label.lower()}. "
        f"The track should begin with a seamless lead-in, evolving naturally from previous activity and ending naturally for the next activity."
    )

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

class ActivityEncoder:
    """Turns (img, stat) → (prompt, hidden-state fp16 CPU)."""
    def __init__(self, root_prompt="workout music", mode="class", device=None):

        self.root_prompt = root_prompt
        self.mode        = mode
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- shared T5 encoder -------------------------------------------------
        # from transformers import AutoTokenizer, AutoModel
        # self.tok = AutoTokenizer.from_pretrained("t5-base")
        # self.t5  = (AutoModel.from_pretrained("t5-base")
        #                         .encoder.eval().to(self.device))

        if mode == "class":
            ckpt = next(MODEL_DIR.glob("cnn_classifier_0513_22_ctxt512_sr64.pth"))
            sample = torch.load(ckpt, map_location="cpu")
            from models.encoder.encoder import SimpleCNN
            self.cls = SimpleCNN(len(activity_labels), num_statics=5, img_size=64).eval()
            self.cls.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else: # SigLIP
            from transformers import AutoProcessor, AutoModel
            self.sig_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.sig_enc  = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
            sig_ckpt = MODEL_DIR / "SigLIP_seglen8.pth"
            self.sig_enc.load_state_dict(torch.load(sig_ckpt, map_location="cpu"), strict=False)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, img, statics, prompt_override=None):
        if prompt_override:
            self.root_prompt = prompt_override

        if self.mode == "class": # classifier
            prompt, label = build_prompt_from_classifier(
                            self.root_prompt, img, statics, self.cls, activity_labels)
            # t = self.tok(prompt, return_tensors="pt").to(self.device)
            # hid = self.t5(**t).last_hidden_state.half().cpu()            # [1,T,768]
            hid = None # not used currently

        else: # embedding
            text_seed = self.tok(f"{self.root_prompt}…", return_tensors="pt").to(self.device)
            txt = self.t5(**text_seed).last_hidden_state.half()    # [1,T,768]

            v_in = self.sig_proc(images=img, return_tensors="pt")
            v   = self.sig_enc.get_image_features(**v_in).half()   # [1,768]

            hid = torch.cat([txt, v[:,None,:]], 1).cpu()                 # [1,T+1,768]
    
        return prompt, label, #hid   # ready to serialise fp16
