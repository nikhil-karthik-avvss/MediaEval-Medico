# train_task2_from_subtask1.py
# Multi-task fine-tuning: Answer + Explanation + Localization (weak supervision)
# Usage: run in same environment you used for Subtask-1 (transformers, datasets, torch, pillow, tqdm, rouge, nltk installed)

import os
import json
import math
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm

from datasets import Dataset as HfDataset
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

# ----------------------------
# Config (edit as needed)
# ----------------------------
DATA_ROOT = "./dataset"         # directory containing images/ and metadata.json
MODEL_NAME = "./model-task-1"  # or your Subtask-1 checkpoint path
OUTPUT_DIR = "./blip-task2-multitask"
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 5e-5
MASK_LOSS_WEIGHT = 1.0         # tune 0.1 - 2.0
SEQ_MAX_LEN = 128
EXPL_MAX_LEN = 80
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ",DEVICE)

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Utility: load metadata.json and build HF Dataset-like list
# ----------------------------
def load_local_metadata(data_root: str, split: str):
    path = Path(data_root) / "metadata.json"
    assert path.exists(), f"metadata.json not found at {path}"
    with open(path, "r") as f:
        meta = json.load(f)
    assert split in meta, f"{split} not in metadata.json keys"
    items = meta[split]
    # ensure each item has img_id and question and answer
    for it in items:
        assert "img_id" in it and "question" in it and "answer" in it
    return items

# ----------------------------
# Custom Dataset (hf_dataset-like wrapper)
# ----------------------------
class LocalKvasirDataset(Dataset):
    def __init__(self, items, images_dir: str, processor: BlipProcessor, seq_max_len=SEQ_MAX_LEN):
        self.items = items
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.seq_max_len = seq_max_len

    def __len__(self):
        return len(self.items)

    def _load_image(self, img_id: str) -> Image.Image:
        p = self.images_dir / f"{img_id}.jpg"
        if not p.exists():
            # fallback to png
            p = self.images_dir / f"{img_id}.png"
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx):
        ex = self.items[idx]
        img = self._load_image(ex["img_id"])
        question = ex["question"]
        # Combined target: answer + explanation placeholder (if explanation exists, include it)
        answer = str(ex.get("answer", "")).strip().lower()
        explanation = ex.get("explanation", "")
        if explanation:
            explanation = str(explanation).strip().lower()
            target_text = f"answer: {answer} </s> explanation: {explanation}"
        else:
            # placeholder; model will generate explanation
            target_text = f"answer: {answer} </s> explanation:"

        # Prepare encoding using processor later in collate to avoid heavy ops here
        return {
            "image": img,
            "question": question,
            "target_text": target_text,
            "img_id": ex["img_id"],
            "answer": answer,
            "raw": ex
        }

# ----------------------------
# Collate: use processor to create labels and inputs
# ----------------------------
def collate_fn(batch, processor: BlipProcessor, device: str = "cpu"):
    images = [b["image"] for b in batch]
    texts = [b["question"] for b in batch]
    targets = [b["target_text"] for b in batch]

    enc = processor(images=images, text=texts, padding="max_length", truncation=True, max_length=SEQ_MAX_LEN, return_tensors="pt")
    with processor.as_target_processor():
        labels = processor(text_target=targets, padding="max_length", truncation=True, max_length=SEQ_MAX_LEN, return_tensors="pt").input_ids

    # replace pad token id with -100 for hf cross-entropy ignore
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch_out = {
        "pixel_values": enc["pixel_values"].to(device),
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
        # keep metadata for visualization later
        "images_pil": images,
        "meta": [{"img_id": b["img_id"], "answer": b["answer"]} for b in batch],
    }
    return batch_out

# ----------------------------
# Attention-rollout -> pseudo mask (helper)
# ----------------------------
def attention_rollout_from_vision(attentions):
    """
    attentions: tuple of tensors, each (batch, heads, seq_len, seq_len)
    returns: cls->patch relevance vector length = num_patches
    """
    # support batch dim =1
    mats = []
    for layer_att in attentions:
        # layer_att shape: (batch, head, seq, seq)
        arr = layer_att[0].detach().cpu()  # (head, seq, seq)
        arr = arr.mean(dim=0)  # average heads -> (seq, seq)
        mats.append(arr.numpy())
    # rollout
    rollout = mats[0] + np.eye(mats[0].shape[0])
    rollout = rollout / rollout.sum(axis=-1, keepdims=True)
    for m in mats[1:]:
        m = m + np.eye(m.shape[0])
        m = m / m.sum(axis=-1, keepdims=True)
        rollout = m @ rollout
    cls2patch = rollout[0, 1:]  # skip cls->cls
    # normalize
    cls2patch = cls2patch - cls2patch.min()
    if cls2patch.max() > 0:
        cls2patch = cls2patch / (cls2patch.max() + 1e-8)
    return cls2patch  # numpy array length = num_patches

def patches_to_grid_mask(cls_attn_vec, grid_size, out_size):
    """
    cls_attn_vec: (P,) vector
    grid_size: int (sqrt(P))
    out_size: (W, H) desired output resolution
    """
    import cv2
    grid = cls_attn_vec.reshape(grid_size, grid_size)
    heat = cv2.resize(grid, out_size, interpolation=cv2.INTER_CUBIC)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat.astype("float32")

# ----------------------------
# Custom model: subclass HF BLIP to add mask head and combined loss
# ----------------------------
'''class BlipForQuestionAnsweringWithMask(BlipForQuestionAnswering):
    def __init__(self, config=None, base_model_name_or_path: Optional[str] = None, mask_feat_size: int = 256):
        if base_model_name_or_path is not None:
            # load pre-trained model via parent constructor
            super().__init__.from_pretrained(base_model_name_or_path)
        else:
            super().__init__(config)
        # hidden size
        hidden = self.config.hidden_size
        # linear maps from patch embeddings -> small feature map
        self.patch2feat = nn.Linear(hidden, mask_feat_size)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_feat_size, mask_feat_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mask_feat_size // 2, 1, kernel_size=1)
        )
        # initialize small head
        nn.init.normal_(self.patch2feat.weight, std=0.02)
        # Keep rest of model weights as is (from parent)'''

class BlipForQuestionAnsweringWithMask(BlipForQuestionAnswering):
    def __init__(self, config=None, base_model_name_or_path: Optional[str] = None, mask_feat_size: int = 256):
        if base_model_name_or_path is not None:
            # Correct way: call class method
            model = BlipForQuestionAnswering.from_pretrained(base_model_name_or_path)
            # copy all weights to self
            self.__dict__.update(model.__dict__)
        else:
            super().__init__(config)
        # now add your mask head
        #hidden = self.config.hidden_size
        hidden = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.vision_config.hidden_size

        self.patch2feat = nn.Linear(hidden, mask_feat_size)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_feat_size, mask_feat_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mask_feat_size // 2, 1, kernel_size=1)
        )
        nn.init.normal_(self.patch2feat.weight, std=0.02)


    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, output_attentions=True, return_dict=True, **kwargs):
        """
        Calls parent forward but extracts vision outputs to compute mask prediction and mask loss (if labels provided).
        Returns a HF-style output; sets loss to combined loss when labels present.
        """
        # we call the parent's forward to get base outputs including loss (lm loss) and vision outputs.
        # Note: use return_dict=True and request vision outputs to include attentions.
        # Some model variants put vision outputs at outputs.vision_model_output or outputs.vision_model
        outputs = super().forward(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=output_attentions, return_dict=return_dict, **kwargs)
        lm_loss = getattr(outputs, "loss", None)

        # Try to extract vision_model outputs (implementation detail differs by HF version)
        vision_out = getattr(outputs, "vision_model_output", None)
        patch_mask_pred = None
        mask_loss = None

        if vision_out is not None and hasattr(vision_out, "last_hidden_state"):
            # last_hidden_state shape: (batch, seq_len, hidden)
            last_hidden = vision_out.last_hidden_state  # tensor
            # remove cls token if present and map to patch features
            if last_hidden.size(1) > 1:
                patch_emb = last_hidden[:, 1:, :]  # (B, P, H)
                B, P, H = patch_emb.shape
                feat = self.patch2feat(patch_emb)  # (B, P, feat)
                # reshape to grid
                grid_size = int(math.sqrt(P))
                feat = feat.permute(0, 2, 1).reshape(B, -1, grid_size, grid_size)  # (B, feat_size, g, g)
                mask_logits = self.mask_conv(feat)  # (B, 1, g, g)
                patch_mask_pred = mask_logits.squeeze(1)  # (B, g, g)

                # If labels present, compute mask pseudo-target from vision_out.attentions (weak supervision)
                if labels is not None and getattr(vision_out, "attentions", None) is not None:
                    # vision_out.attentions is tuple(layers) each (B, heads, seq, seq)
                    attns = vision_out.attentions  # tuple
                    # compute cls->patch vector for each batch element
                    targets = []
                    for b in range(B):
                        clsvec = attention_rollout_from_vision([a[b:b+1] for a in attns])  # length P
                        # reshape/resize to patch_mask_pred spatial size
                        target_grid = torch.tensor(clsvec, dtype=patch_mask_pred.dtype, device=patch_mask_pred.device).reshape(grid_size, grid_size)
                        # ensure same shape
                        if target_grid.shape != patch_mask_pred[b].shape:
                            target_grid = F.interpolate(target_grid.unsqueeze(0).unsqueeze(0), size=patch_mask_pred.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                        targets.append(target_grid)
                    target_stack = torch.stack(targets, dim=0)  # (B, g, g)
                    # compute BCE with logits (target in [0,1])
                    mask_loss = F.binary_cross_entropy_with_logits(patch_mask_pred, target_stack)

        # combine losses
        total_loss = None
        if lm_loss is not None:
            if mask_loss is not None:
                total_loss = lm_loss + MASK_LOSS_WEIGHT * mask_loss
            else:
                total_loss = lm_loss
        # attach combined loss into outputs and return
        if total_loss is not None:
            outputs.loss = total_loss

        # Also attach patch_mask_pred for inference convenience
        outputs.patch_mask_pred = patch_mask_pred if patch_mask_pred is not None else None
        outputs.mask_loss = mask_loss if mask_loss is not None else None
        return outputs

# ----------------------------
# Inference helpers: parse combined text, generate, compute confidence
# ----------------------------
def parse_answer_explanation(text: str):
    txt = text.strip()
    lower = txt.lower()
    if "explanation:" in lower:
        idx = lower.index("explanation:")
        ans = txt[:idx].replace("answer:", "").strip()
        expl = txt[idx + len("explanation:"):].strip()
        return ans, expl
    if "</s>" in txt:
        a, e = txt.split("</s>", 1)
        return a.replace("answer:", "").strip(), e.strip()
    return txt.strip(), ""

def generate_with_confidence(model: BlipForQuestionAnsweringWithMask, processor: BlipProcessor, image: Image.Image, question: str, max_new_tokens=64, device: str = DEVICE):
    model.to(device)
    model.eval()
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True)
    txt = processor.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
    # compute mean token probability
    token_probs = []
    for step, logits in enumerate(out.scores):
        token_id = int(out.sequences[0, 1 + step])  # typically sequences contain bos token at 0
        probs = F.softmax(logits[0], dim=-1)
        token_probs.append(float(probs[token_id].cpu().item()))
    conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.0
    ans, expl = parse_answer_explanation(txt)
    return ans, expl, conf, out

def upsample_patch_mask_to_image(patch_mask_logits, image_size):
    # patch_mask_logits: torch tensor (g, g) or numpy
    import cv2
    if isinstance(patch_mask_logits, torch.Tensor):
        arr = patch_mask_logits.detach().cpu().numpy()
    else:
        arr = patch_mask_logits
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    prob = 1.0 / (1.0 + np.exp(-arr))
    H, W = image_size[1], image_size[0]
    heat = cv2.resize(prob, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


from functools import partial

def data_collator(batch,processor,DEVICE):
    return collate_fn(batch, processor=processor, device=DEVICE)


# ----------------------------
# Main: prepare data, model, trainer, train
# ----------------------------
def main():
    # load metadata
    train_items = load_local_metadata(DATA_ROOT, "train")
    test_items = load_local_metadata(DATA_ROOT, "test")

    print(f"Train items: {len(train_items)}, Test items: {len(test_items)}")

    # processor + model
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    # instantiate our custom model using pretrained weights
    model = BlipForQuestionAnsweringWithMask(base_model_name_or_path=MODEL_NAME)

    # build dataset objects
    train_ds = LocalKvasirDataset(train_items, images_dir=os.path.join(DATA_ROOT, "images"), processor=processor)
    test_ds = LocalKvasirDataset(test_items, images_dir=os.path.join(DATA_ROOT, "images"), processor=processor)

    # we will pass datasets to Trainer with data collator-like function by wrapping them into HF Dataset-like objects
    # But Trainer expects a dataset returning dict[str, Tensor], easier to pass through Trainer if we create small wrappers using map.
    # Simpler: create torch.utils.data.DataLoader via Trainer? Trainer can accept PyTorch dataset. We'll use Trainer but provide our collator via data_collator param.
    

    # TrainingArguments similar to your original script
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        dataloader_num_workers=4,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
    )

    # Build Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=partial(data_collator,processor=processor,DEVICE=DEVICE),
        tokenizer=processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,  # used for batching shape but not mandatory
    )

    # Train
    trainer.train()

    # Save final weights and processor
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    # ----------------------------
    # Inference on test -> create submission-like JSONL with visualizations
    # ----------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    vis_dir = os.path.join(OUTPUT_DIR, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    entries = []
    for ex in tqdm(test_items, desc="Infer test"):
        img = Image.open(os.path.join(DATA_ROOT, "images", f"{ex['img_id']}.jpg")).convert("RGB")
        q = ex["question"]
        ans, expl, conf, gen_out = generate_with_confidence(model, processor, img, q)
        # forward one pass to get patch_mask_pred
        with torch.no_grad():
            enc = processor(images=img, text=q, return_tensors="pt").to(DEVICE)
            forward_out = model(pixel_values=enc["pixel_values"], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_attentions=True)
            patch_mask_pred = getattr(forward_out, "patch_mask_pred", None)
        visual_expls = []
        if patch_mask_pred is not None:
            heat = upsample_patch_mask_to_image(patch_mask_pred[0], img.size)  # (H,W)
            # save overlay
            fig_path = os.path.join(vis_dir, f"{ex['img_id']}_heat.png")
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.imshow(heat, cmap="jet", alpha=0.4)
            plt.axis("off")
            plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            visual_expls.append({"type":"heatmap", "data": fig_path, "description": "learned localization heatmap"})

        entry = {
            "img_id": ex["img_id"],
            "question": q,
            "answer": ans,
            "textual_explanation": expl,
            "visual_explanation": visual_expls,
            "confidence_score": float(conf)
        }
        entries.append(entry)

    out_file = os.path.join(OUTPUT_DIR, "submission_task2.jsonl")
    with open(out_file, "w") as wf:
        for e in entries:
            wf.write(json.dumps(e) + "\n")
    print("Wrote submission file:", out_file)

if __name__ == "__main__":
    main()
