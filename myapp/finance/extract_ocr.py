import os
import json
import re
from pathlib import Path
from datasets import Dataset
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification, Trainer, TrainingArguments
import torch
import pytesseract
from PIL import Image
import numpy as np

# Define labels
label_list = ["O", "B-ITEM", "I-ITEM", "B-TOTAL", "I-TOTAL"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Initialize fast tokenizer and model
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label
)

# Custom function to parse malformed JSON-like data
def parse_malformed_json(raw_data):
    try:
        raw_data = raw_data.replace("'", '"')
        raw_data = re.sub(r',\s*}', '}', raw_data)
        raw_data = re.sub(r',\s*\]', ']', raw_data)
        return json.loads(raw_data)
    except json.JSONDecodeError as e:
        label_data = {}
        pattern = r'"([^"]+)":\s*"([^"]+)"'
        matches = re.findall(pattern, raw_data)
        for key, value in matches:
            label_data[key] = value
        return label_data

# Preprocessing function for SROIE dataset
def preprocess_sroie(data_dir):
    data_dir = Path(data_dir) / "SROIE2019"
    image_dir = data_dir / "train" / "img"
    ocr_dir = data_dir / "train" / "box"
    label_dir = data_dir / "train" / "entities"

    print(f"Image directory: {image_dir}")
    print(f"OCR directory: {ocr_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Files in image_dir: {[f.name for f in image_dir.glob('*')][:5]}")
    print(f"Files in ocr_dir: {[f.name for f in ocr_dir.glob('*')][:5]}")
    print(f"Files in label_dir: {[f.name for f in label_dir.glob('*')][:5]}")

    # Collect file IDs from all directories
    image_ids = {f.stem for f in image_dir.glob("*.jpg")}
    ocr_ids = {f.stem for f in ocr_dir.glob("*.txt")}
    label_ids = {f.stem for f in label_dir.glob("*.txt")}

    # Find common IDs
    common_ids = image_ids.intersection(ocr_ids).intersection(label_ids)
    print(f"Number of common IDs: {len(common_ids)}")
    if not common_ids:
        raise ValueError("No matching files found across image, OCR, and label directories.")

    data = {"words": [], "bboxes": [], "ner_tags": [], "image_path": []}

    for file_id in common_ids:
        image_path = image_dir / f"{file_id}.jpg"

        # Load OCR data
        ocr_path = ocr_dir / f"{file_id}.txt"
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_data = f.readlines()

        words = []
        bboxes = []
        for line in ocr_data:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[5])
                text = ",".join(parts[8:]).strip()
                if not text:
                    continue
                img = Image.open(image_path)
                width, height = img.size
                normalized_bbox = [
                    int(1000 * x1 / width),
                    int(1000 * y1 / height),
                    int(1000 * x2 / width),
                    int(1000 * y2 / height)
                ]
                words.append(text)
                bboxes.append(normalized_bbox)
            except (ValueError, IndexError) as e:
                print(f"Error processing OCR line in {file_id}: {line}, skipping...")
                continue

        # Load ground truth labels
        label_path = label_dir / f"{file_id}.txt"
        with open(label_path, "r", encoding="utf-8") as f:
            label_data_raw = f.read()
            try:
                label_data = parse_malformed_json(label_data_raw)
            except Exception as e:
                print(f"Error parsing label file for {file_id}: {e}, skipping...")
                continue

        # Align words with labels
        ner_tags = ["O"] * len(words)
        total_value = label_data.get("total", "").strip()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if total_value and word_lower in total_value.lower():
                ner_tags[i] = "B-TOTAL" if i == 0 or ner_tags[i-1] == "O" else "I-TOTAL"
            elif not any(keyword in word_lower for keyword in ["total", "subtotal", "tax"]):
                ner_tags[i] = "B-ITEM" if i == 0 or ner_tags[i-1] == "O" else "I-ITEM"

        data["words"].append(words)
        data["bboxes"].append(bboxes)
        data["ner_tags"].append([label2id[tag] for tag in ner_tags])
        data["image_path"].append(str(image_path))

    if not data["words"]:
        raise ValueError("No valid data found after preprocessing. Check dataset paths and files.")

    dataset = Dataset.from_dict(data)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample example: {dataset[0]}")
    return dataset

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )

    labels = []
    bboxes = []
    for i in range(len(examples["words"])):
        word_ids_list = examples["words"][i]
        ner_tags = examples["ner_tags"][i]
        label_ids = []
        bbox_list = []
        word_idx = -1
        for word_idx_in_tokens in tokenized_inputs.word_ids(batch_index=i):  # Fixed parameter name
            if word_idx_in_tokens is None:
                label_ids.append(-100)
                bbox_list.append([0, 0, 0, 0])
            else:
                if word_idx + 1 < len(word_ids_list):
                    word_idx += 1
                    label_ids.append(ner_tags[word_idx])
                    bbox_list.append(examples["bboxes"][i][word_idx])
                else:
                    label_ids.append(-100)
                    bbox_list.append([0, 0, 0, 0])
        labels.append(label_ids)
        bboxes.append(bbox_list)

    tokenized_inputs["labels"] = torch.tensor(labels)
    tokenized_inputs["bbox"] = torch.tensor(bboxes)
    return tokenized_inputs

# Load and preprocess dataset
data_dir = "/kaggle/input/sroie-datasetv2"
dataset = preprocess_sroie(data_dir)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split into train and test
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./layoutlm-finetuned-sroie",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # Disable reporting to wandb
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./layoutlm-finetuned-sroie")
tokenizer.save_pretrained("./layoutlm-finetuned-sroie")

# Inference on a new receipt
def extract_entities(image_path):
    img = Image.open(image_path)
    ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    bboxes = []
    width, height = img.size
    for i in range(len(ocr_result["text"])):
        text = ocr_result["text"][i].strip()
        if not text:
            continue
        x, y, w, h = (
            ocr_result["left"][i],
            ocr_result["top"][i],
            ocr_result["width"][i],
            ocr_result["height"][i],
        )
        bbox = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(text)
        bboxes.append(bbox)

    encoding = tokenizer(
        words,
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )

    token_boxes = []
    word_idx = -1
    for idx in encoding.word_ids(batch_index=0):  # Fixed parameter name
        if idx is None:
            token_boxes.append([0, 0, 0, 0])
        else:
            word_idx += 1
            if word_idx < len(bboxes):
                token_boxes.append(bboxes[word_idx])
            else:
                token_boxes.append([0, 0, 0, 0])

    encoding["bbox"] = torch.tensor([token_boxes])

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)

    items = []
    current_item = []
    totals = []
    current_total = []
    word_idx = -1
    for idx, (word_id, label_id) in enumerate(zip(encoding.word_ids(0), predictions[0].tolist())):
        if word_id is None:
            continue
        word_idx += 1
        if word_idx >= len(words):
            break
        word = words[word_idx]
        label = id2label[label_id]

        if label == "B-ITEM":
            if current_item:
                items.append(" ".join(current_item))
            current_item = [word]
        elif label == "I-ITEM":
            current_item.append(word)
        elif label == "B-TOTAL":
            if current_total:
                totals.append(" ".join(current_total))
            current_total = [word]
        elif label == "I-TOTAL":
            current_total.append(word)
        else:
            if current_item:
                items.append(" ".join(current_item))
                current_item = []
            if current_total:
                totals.append(" ".join(current_total))
                current_total = []

    if current_item:
        items.append(" ".join(current_item))
    if current_total:
        totals.append(" ".join(current_total))

    return {"items": items, "totals": totals}

# Test inference
new_receipt_path = '/content/toys.jpg'
result = extract_entities(new_receipt_path)
print(result)
