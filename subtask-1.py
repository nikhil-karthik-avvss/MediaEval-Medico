import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_from_disk
from transformers import BlipForQuestionAnswering, BlipProcessor, Trainer, TrainingArguments
from tqdm import tqdm
import pandas as pd

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import matplotlib.pyplot as plt


# --------------------
# Custom Dataset
# --------------------
class KvasirVQADataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.ds = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]

        # ✅ Use cached local image path
        img_path = example.get("local_image", None)
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # fallback to original HF "image" field
            image = example["image"]

        encoding = self.processor(
            images=image,
            text=example["question"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # normalize answers
        answer_text = str(example["answer"]).strip().lower()

        label_encoding = self.processor.tokenizer(
            answer_text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt"
        )

        encoding["labels"] = label_encoding.input_ids.squeeze()
        return {k: v.squeeze() for k, v in encoding.items()}


def main():
    # --------------------
    # 1. Load dataset (LOCAL)
    # --------------------
    dataset = load_from_disk("./kvasir_vqa_local_cached")  # ✅ use cached version
    train_data = dataset["train"]
    test_data = dataset["test"]

    print("Train size:", len(train_data))
    print("Test size :", len(test_data))

    # --------------------
    # 2. BLIP model + processor
    # --------------------
    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    model.to(device)

    train_dataset = KvasirVQADataset(train_data, processor)
    test_dataset = KvasirVQADataset(test_data, processor)

    # --------------------
    # 3. Training setup
    # --------------------
    '''training_args = TrainingArguments(
        output_dir="./blip-kvasir-x1",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        learning_rate=5e-5,
        eval_strategy="steps",  # ✅ was `eval_strategy` (typo)
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        logging_dir="./logs",
        fp16=True,
        dataloader_num_workers=4,
        save_total_limit=2,
        report_to="none"
    )'''

    training_args = TrainingArguments(
        output_dir="./blip-kvasir-x1",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_dir="./logs",
        fp16=True,
        dataloader_num_workers=4,
        save_total_limit=2,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,  # ✅ use processor instead of tokenizer
    )

    # --------------------
    # 4. Train
    # --------------------
    trainer.train()

    # Save model
    model.save_pretrained("./blip-kvasir-x1-final")
    processor.save_pretrained("./blip-kvasir-x1-final")

    # --------------------
    # 5. Inference on test set
    # --------------------
    predictions, references = [], []
    for i in tqdm(range(len(test_data)), desc="Evaluating"):
        example = test_data[i]

        img_path = example.get("local_image", None)
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = example["image"]

        question = example["question"]
        gt_answer = str(example["answer"]).strip().lower()

        inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_length=30)

        pred_answer = processor.tokenizer.decode(out[0], skip_special_tokens=True).lower().strip()

        predictions.append(pred_answer)
        references.append(gt_answer)

    # Save predictions
    df_eval = pd.DataFrame({
        "question": [ex["question"] for ex in test_data],
        "ground_truth": references,
        "prediction": predictions
    })
    df_eval.to_csv("kvasir_x1_predictions.csv", index=False)

    # --------------------
    # 6. Evaluation Metrics
    # --------------------
    smooth_fn = SmoothingFunction().method1
    rouge = Rouge()

    bleu_scores, meteor_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], [], []

    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        meteor_scores.append(meteor_score([ref], pred))

        rouge_scores = rouge.get_scores(pred, ref)[0]
        rouge1_scores.append(rouge_scores["rouge-1"]["f"])
        rouge2_scores.append(rouge_scores["rouge-2"]["f"])
        rougeL_scores.append(rouge_scores["rouge-l"]["f"])

    print("\n===== Evaluation Results (Test set) =====")
    print(f"BLEU  (avg): {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"METEOR(avg): {sum(meteor_scores)/len(meteor_scores):.4f}")
    print(f"ROUGE-1(avg): {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-2(avg): {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"ROUGE-L(avg): {sum(rougeL_scores)/len(rougeL_scores):.4f}")

    # --------------------
    # 7. Visualization
    # --------------------
    metrics = {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "METEOR": sum(meteor_scores)/len(meteor_scores),
        "ROUGE-1": sum(rouge1_scores)/len(rouge1_scores),
        "ROUGE-2": sum(rouge2_scores)/len(rouge2_scores),
        "ROUGE-L": sum(rougeL_scores)/len(rougeL_scores)
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color="skyblue")
    plt.title("Evaluation Metrics on Kvasir-VQA-x1 Test Set")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig("evaluation_metrics.png")
    plt.show()

    sample_df = df_eval.sample(10)
    print("\n===== Sample Predictions =====")
    print(sample_df)


if __name__ == "__main__":
    main()
