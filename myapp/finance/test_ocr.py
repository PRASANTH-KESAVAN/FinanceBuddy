# import os
# import re
# import torch
# import pytesseract
# import easyocr
# import cv2
# import numpy as np
# from PIL import Image
# from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
# from pathlib import Path

# # Suppress TensorFlow oneDNN warnings
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# # Define labels (same as during training)
# label_list = ["O", "B-ITEM", "I-ITEM", "B-TOTAL", "I-TOTAL"]
# label2id = {label: idx for idx, label in enumerate(label_list)}
# id2label = {idx: label for label, idx in label2id.items()}

# # Load trained model and tokenizer (use absolute path)
# model_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/layoutlm-finetuned-sroie"
# try:
#     tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
#     model = LayoutLMForTokenClassification.from_pretrained(model_path)
# except Exception as e:
#     print(f"Error loading model or tokenizer from {model_path}: {str(e)}")
#     exit(1)

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # LayoutLM inference function with refined product name extraction
# def extract_entities_layoutlm(image_path):
#     try:
#         # Load image
#         img = Image.open(image_path)
#         width, height = img.size

#         # Perform OCR using pytesseract
#         ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

#         words = []
#         bboxes = []
#         for i in range(len(ocr_result["text"])):
#             text = ocr_result["text"][i].strip()
#             if not text:
#                 continue
#             x, y, w, h = (
#                 ocr_result["left"][i],
#                 ocr_result["top"][i],
#                 ocr_result["width"][i],
#                 ocr_result["height"][i],
#             )
#             # Normalize bounding boxes to [0, 1000]
#             bbox = [
#                 int(1000 * x / width),
#                 int(1000 * y / height),
#                 int(1000 * (x + w) / width),
#                 int(1000 * (y + h) / height),
#             ]
#             words.append(text)
#             bboxes.append(bbox)

#         if not words:
#             return {"items": [], "totals": [], "error": "No text detected by OCR"}

#         # Tokenize inputs
#         encoding = tokenizer(
#             words,
#             truncation=True,
#             padding="max_length",
#             max_length=512,
#             is_split_into_words=True,
#             return_tensors="pt"
#         )

#         # Prepare bounding boxes for tokens
#         token_boxes = []
#         word_idx = -1
#         for idx in encoding.word_ids(batch_index=0):
#             if idx is None:
#                 token_boxes.append([0, 0, 0, 0])
#             else:
#                 word_idx += 1
#                 if word_idx < len(bboxes):
#                     token_boxes.append(bboxes[word_idx])
#                 else:
#                     token_boxes.append([0, 0, 0, 0])

#         encoding["bbox"] = torch.tensor([token_boxes])

#         # Run model inference
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**encoding)
#             predictions = torch.argmax(outputs.logits, dim=-1)

#         # Extract entities with product name filtering
#         items = []
#         current_item = []
#         totals = []
#         current_total = []
#         word_idx = -1
#         print("Debugging Predictions:")
#         for idx, (word_id, label_id) in enumerate(zip(encoding.word_ids(0), predictions[0].tolist())):
#             if word_id is None:
#                 continue
#             word_idx += 1
#             if word_idx >= len(words):
#                 break
#             word = words[word_idx]
#             label = id2label[label_id]
#             print(f"Word: {word}, Label: {label}")

#             # Logic to extract product names
#             if label == "B-ITEM":
#                 if current_item:
#                     item_text = " ".join(current_item)
#                     # Filter: include only valid product names
#                     if (
#                         len(item_text) > 2  # Short enough to be a product name
#                         and not any(keyword in item_text.lower() for keyword in [
#                             "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp"
#                         ])
#                         and not re.match(r"^\d+(\.\d+)?$", item_text)  # Exclude pure numbers/amounts
#                         and re.search(r"[A-Za-z]", item_text)  # Must contain letters
#                     ):
#                         items.append(item_text)
#                 current_item = [word]
#             elif label == "I-ITEM":
#                 current_item.append(word)
#             elif label == "B-TOTAL":
#                 if current_total:
#                     totals.append(" ".join(current_total))
#                 current_total = [word]
#             elif label == "I-TOTAL":
#                 current_total.append(word)
#             else:
#                 if current_item:
#                     item_text = " ".join(current_item)
#                     if (
#                         len(item_text) > 2
#                         and not any(keyword in item_text.lower() for keyword in [
#                             "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp"
#                         ])
#                         and not re.match(r"^\d+(\.\d+)?$", item_text)
#                         and re.search(r"[A-Za-z]", item_text)
#                     ):
#                         items.append(item_text)
#                     current_item = []
#                 if current_total:
#                     totals.append(" ".join(current_total))
#                     current_total = []

#         if current_item:
#             item_text = " ".join(current_item)
#             if (
#                 len(item_text) > 2
#                 and not any(keyword in item_text.lower() for keyword in [
#                     "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp"
#                 ])
#                 and not re.match(r"^\d+(\.\d+)?$", item_text)
#                 and re.search(r"[A-Za-z]", item_text)
#             ):
#                 items.append(item_text)

#         if current_total:
#             totals.append(" ".join(current_total))

#         return {"items": items, "totals": totals}

#     except Exception as e:
#         return {"items": [], "totals": [], "error": str(e)}

# # EasyOCR extraction function with product name description
# def extract_receipt_data_easyocr(image_path):
#     try:
#         # Load image
#         img = cv2.imread(image_path)
#         if img is None:
#             return {"status": "error", "message": "Failed to load image"}

#         # Run EasyOCR
#         results = reader.readtext(img, detail=0)
#         text = "\n".join(results)

#         # Extract amounts (e.g., 12.34)
#         amounts = re.findall(r"\d+\.\d{2}", text)
#         total_amount = max([float(a) for a in amounts]) if amounts else 0.0

#         # Get items from LayoutLM
#         layoutlm_result = extract_entities_layoutlm(image_path)
#         items = layoutlm_result.get("items", [])

#         # Create description from product names only
#         description = ', '.join(items[:5]) if items else "No products detected"

#         return {
#             "status": "success",
#             "amount": total_amount,
#             "items": items,
#             "description": description,
#             "full_text": text,
#             "totals": layoutlm_result.get("totals", []),
#             "error": layoutlm_result.get("error", None)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # Define the path to the receipt image (absolute)
# new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/snacks.jpg"
# new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/GroceryStoreReceipt.jpg"
# new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/toys.jpg"



# # Check if the image exists
# if not os.path.exists(new_receipt_path):
#     print(f"Image not found at {new_receipt_path}")
#     print(f"Resolved path: {os.path.abspath(new_receipt_path)}")
# else:
#     # Run inference
#     result = extract_receipt_data_easyocr(new_receipt_path)
#     print("Extraction Result:")
#     print(f"Status: {result['status']}")
#     if result["status"] == "success":
#         print(f"Total Amount: ${result['amount']:.2f}")
#         print(f"Items: {result['items']}")
#         print(f"Totals: {result['totals']}")
#         print(f"Description: {result['description']}")
#         print(f"Full Text:\n{result['full_text']}")
#     else:
#         print(f"Error: {result['message']}")
#     if result.get("error"):
#         print(f"LayoutLM Error: {result['error']}")






# import os
# import re
# import torch
# import pytesseract
# import easyocr
# import cv2
# import numpy as np
# from PIL import Image
# from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
# from pathlib import Path
# import warnings

# # Suppress TensorFlow oneDNN warnings
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# # Suppress PyTorch pin_memory warning
# warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")

# # Define labels (same as during training)
# label_list = ["O", "B-ITEM", "I-ITEM", "B-TOTAL", "I-TOTAL"]
# label2id = {label: idx for idx, label in enumerate(label_list)}
# id2label = {idx: label for label, idx in label2id.items()}

# # Load trained model and tokenizer (use absolute path for reliability)
# model_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/layoutlm-finetuned-sroie"
# try:
#     tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
#     model = LayoutLMForTokenClassification.from_pretrained(model_path)
# except Exception as e:
#     print(f"Error loading model or tokenizer from {model_path}: {str(e)}")
#     exit(1)

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # LayoutLM inference function (unchanged)
# def extract_entities_layoutlm(image_path):
#     try:
#         # Load image
#         img = Image.open(image_path)
#         width, height = img.size

#         # Perform OCR using pytesseract
#         ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

#         words = []
#         bboxes = []
#         for i in range(len(ocr_result["text"])):
#             text = ocr_result["text"][i].strip()
#             if not text:
#                 continue
#             x, y, w, h = (
#                 ocr_result["left"][i],
#                 ocr_result["top"][i],
#                 ocr_result["width"][i],
#                 ocr_result["height"][i],
#             )
#             # Normalize bounding boxes to [0, 1000]
#             bbox = [
#                 int(1000 * x / width),
#                 int(1000 * y / height),
#                 int(1000 * (x + w) / width),
#                 int(1000 * (y + h) / height),
#             ]
#             words.append(text)
#             bboxes.append(bbox)

#         if not words:
#             return {"items": [], "totals": [], "error": "No text detected by OCR"}

#         # Tokenize inputs
#         encoding = tokenizer(
#             words,
#             truncation=True,
#             padding="max_length",
#             max_length=512,
#             is_split_into_words=True,
#             return_tensors="pt"
#         )

#         # Prepare bounding boxes for tokens
#         token_boxes = []
#         word_idx = -1
#         for idx in encoding.word_ids(batch_index=0):
#             if idx is None:
#                 token_boxes.append([0, 0, 0, 0])
#             else:
#                 word_idx += 1
#                 if word_idx < len(bboxes):
#                     token_boxes.append(bboxes[word_idx])
#                 else:
#                     token_boxes.append([0, 0, 0, 0])

#         encoding["bbox"] = torch.tensor([token_boxes])

#         # Run model inference
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**encoding)
#             predictions = torch.argmax(outputs.logits, dim=-1)

#         # Extract entities
#         items = []
#         current_item = []
#         totals = []
#         current_total = []
#         word_idx = -1
#         print("Debugging LayoutLM Predictions:")
#         for idx, (word_id, label_id) in enumerate(zip(encoding.word_ids(0), predictions[0].tolist())):
#             if word_id is None:
#                 continue
#             word_idx += 1
#             if word_idx >= len(words):
#                 break
#             word = words[word_idx]
#             label = id2label[label_id]
#             print(f"Word: {word}, Label: {label}")

#             if label == "B-ITEM":
#                 if current_item:
#                     items.append(" ".join(current_item))
#                 current_item = [word]
#             elif label == "I-ITEM":
#                 current_item.append(word)
#             elif label == "B-TOTAL":
#                 if current_total:
#                     totals.append(" ".join(current_total))
#                 current_total = [word]
#             elif label == "I-TOTAL":
#                 current_total.append(word)
#             else:
#                 if current_item:
#                     items.append(" ".join(current_item))
#                     current_item = []
#                 if current_total:
#                     totals.append(" ".join(current_total))
#                     current_total = []

#         if current_item:
#             items.append(" ".join(current_item))
#         if current_total:
#             totals.append(" ".join(current_total))

#         return {"items": items, "totals": totals}

#     except Exception as e:
#         return {"items": [], "totals": [], "error": str(e)}

# # EasyOCR extraction function with refined product name extraction from full_text
# def extract_receipt_data_easyocr(image_path):
#     try:
#         # Load image
#         img = cv2.imread(image_path)
#         if img is None:
#             return {"status": "error", "message": "Failed to load image"}

#         # Run EasyOCR
#         results = reader.readtext(img, detail=0)
#         text = "\n".join(results)

#         # Extract amounts (e.g., 12.34)
#         amounts = re.findall(r"\d+\.\d{2}", text)
#         total_amount = max([float(a) for a in amounts]) if amounts else 0.0

#         # Get items and totals from LayoutLM
#         layoutlm_result = extract_entities_layoutlm(image_path)
#         items = layoutlm_result.get("items", [])

#         # Extract product names from full_text
#         product_names = []
#         lines = text.split("\n")
#         in_item_section = False
#         print("Debugging full_text Product Extraction:")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if not line:
#                 continue
#             # Detect start of itemized section
#             if any(keyword in line.lower() for keyword in ["product", "mrp", "item"]):
#                 in_item_section = True
#                 print(f"Start of item section: {line}")
#                 continue
#             # Stop at total or footer
#             if any(keyword in line.lower() for keyword in ["total", "net amount", "paymode", "gpay", "cashier"]):
#                 in_item_section = False
#                 print(f"End of item section: {line}")
#                 continue
#             if not in_item_section:
#                 print(f"Skipped: {line} (outside item section)")
#                 continue
#             # Get the next line (if available) to check for price/quantity
#             next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
#             # Debug the next-line check
#             price_match = re.match(r"^\d+(\.\d+)?\s*\.?$", next_line)
#             qty_match = re.match(r"^\d+$", next_line)
#             print(f"Next line check for '{line}': next_line='{next_line}', price_match={price_match}, qty_match={qty_match}")
#             # Filter product names
#             is_product = (
#                 len(line) > 2  # Short enough to be a product name
#                 and len(line) <= 50  # Avoid long strings
#                 and not any(keyword in line.lower() for keyword in [
#                     "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp", "sri", "golden", "cholamb", "thanks", "cash", "order", "available", "id", "no:", "main", "pate", "edu", "rumullaivoyal", "thi", "for comming", "paymode", "product", "mrp", "amt", "off", "weight", "nateso", "atxpms", "bi17", "aty"
#                 ])
#                 and not re.match(r"^\d+(\.\d+)?$", line)  # Exclude pure numbers/amounts
#                 and not re.match(r"^\d+$", line)  # Exclude pure integers
#                 and re.match(r"^[A-Za-z\s][A-Za-z0-9\s]*$", line)  # Starts with letter, allows numbers
#                 and (price_match or qty_match)  # Followed by price/quantity
#             )
#             if is_product:
#                 product_names.append(line)
#                 print(f"Accepted: {line} (followed by: {next_line})")
#             else:
#                 reason = (
#                     "too short" if len(line) <= 2 else
#                     "too long" if len(line) > 50 else
#                     "contains excluded keyword" if any(keyword in line.lower() for keyword in [
#                         "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp", "sri", "golden", "cholamb", "thanks", "cash", "order", "available", "id", "no:", "main", "pate", "edu", "rumullaivoyal", "thi", "for comming", "paymode", "product", "mrp", "amt", "off", "weight", "nateso", "atxpms", "bi17", "aty"
#                     ]) else
#                     "pure number/amount" if re.match(r"^\d+(\.\d+)?$", line) or re.match(r"^\d+$", line) else
#                     "invalid format" if not re.match(r"^[A-Za-z\s][A-Za-z0-9\s]*$", line) else
#                     "not followed by price/quantity" if not (price_match or qty_match) else
#                     "unknown"
#                 )
#                 print(f"Rejected: {line} (reason: {reason})")

#         # Create description from product names
#         description = ', '.join(product_names[:5]) if product_names else "No products detected"
#         print(f"Final Product Names: {product_names}")

#         return {
#             "status": "success",
#             "amount": total_amount, ####this one works
#             "items": items,  # From LayoutLM
#             "description": description,  # From full_text  # this one works too
#             "full_text": text,
#             "totals": layoutlm_result.get("totals", []),
#             "error": layoutlm_result.get("error", None)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # Define the path to the receipt image (absolute)
# # new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/snacks.jpg"
# # new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/GroceryStoreReceipt.jpg"
# # new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/toys.jpg"
# new_receipt_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/bill3.png"


# # Check if the image exists
# if not os.path.exists(new_receipt_path):
#     print(f"Image not found at {new_receipt_path}")
#     print(f"Resolved path: {os.path.abspath(new_receipt_path)}")
# else:
#     # Run inference
#     result = extract_receipt_data_easyocr(new_receipt_path)
#     print("Extraction Result:")
#     print(f"Status: {result['status']}")
#     if result["status"] == "success":
#         print(f"Total Amount: ${result['amount']:.2f}")
#         print(f"Items (LayoutLM): {result['items']}")
#         print(f"Totals (LayoutLM): {result['totals']}")
#         print(f"Description (full_text): {result['description']}")
#         print(f"Full Text:\n{result['full_text']}")
#     else:
#         print(f"Error: {result['message']}")
#     if result.get("error"):
#         print(f"LayoutLM Error: {result['error']}")



import os
# Set environment variable before any imports that might load TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import torch
import pytesseract
import easyocr
import cv2
import numpy as np
from PIL import Image
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from pathlib import Path
import warnings
import glob

# Suppress PyTorch pin_memory warning
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")

# Define labels (same as during training)
label_list = ["O", "B-ITEM", "I-ITEM", "B-TOTAL", "I-TOTAL"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Load trained model and tokenizer (use absolute path for reliability)
model_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/layoutlm-finetuned-sroie"
try:
    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
    model = LayoutLMForTokenClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or tokenizer from {model_path}: {str(e)}")
    exit(1)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# LayoutLM inference function (unchanged)
def extract_entities_layoutlm(image_path):
    try:
        # Load image
        img = Image.open(image_path)
        width, height = img.size

        # Perform OCR using pytesseract
        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words = []
        bboxes = []
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
            # Normalize bounding boxes to [0, 1000]
            bbox = [
                int(1000 * x / width),
                int(1000 * y / height),
                int(1000 * (x + w) / width),
                int(1000 * (y + h) / height),
            ]
            words.append(text)
            bboxes.append(bbox)

        if not words:
            return {"items": [], "totals": [], "error": "No text detected by OCR"}

        # Tokenize inputs
        encoding = tokenizer(
            words,
            truncation=True,
            padding="max_length",
            max_length=512,
            is_split_into_words=True,
            return_tensors="pt"
        )

        # Prepare bounding boxes for tokens
        token_boxes = []
        word_idx = -1
        for idx in encoding.word_ids(batch_index=0):
            if idx is None:
                token_boxes.append([0, 0, 0, 0])
            else:
                word_idx += 1
                if word_idx < len(bboxes):
                    token_boxes.append(bboxes[word_idx])
                else:
                    token_boxes.append([0, 0, 0, 0])

        encoding["bbox"] = torch.tensor([token_boxes])

        # Run model inference
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Extract entities
        items = []
        current_item = []
        totals = []
        current_total = []
        word_idx = -1
        print("Debugging LayoutLM Predictions:")
        for idx, (word_id, label_id) in enumerate(zip(encoding.word_ids(0), predictions[0].tolist())):
            if word_id is None:
                continue
            word_idx += 1
            if word_idx >= len(words):
                break
            word = words[word_idx]
            label = id2label[label_id]
            print(f"Word: {word}, Label: {label}")

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

    except Exception as e:
        return {"items": [], "totals": [], "error": str(e)}

# EasyOCR extraction function with refined product name extraction from full_text
def extract_receipt_data_easyocr(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Failed to load image"}

        # Run EasyOCR
        results = reader.readtext(img, detail=0)
        text = "\n".join(results)

        # Extract amounts (e.g., 12.34, $12.34, 12,34)
        amounts = re.findall(r"(?:\$\s*)?\d+(?:[,.]\d{2})?", text)
        total_amount = max([float(a.replace("$", "").replace(",", ".").replace(" ", "")) for a in amounts]) if amounts else 0.0

        # Get items and totals from LayoutLM
        layoutlm_result = extract_entities_layoutlm(image_path)
        items = layoutlm_result.get("items", [])

        # Extract product names from full_text
        product_names = []
        lines = text.split("\n")
        in_item_section = False
        print("Debugging full_text Product Extraction:")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Detect start of itemized section
            if any(keyword in line.lower() for keyword in ["description", "product", "mrp", "item(s)"]):
                in_item_section = True
                print(f"Start of item section: {line}")
                continue
            # Stop at total, payment, or footer
            if any(keyword in line.lower() for keyword in ["total", "subtotal", "net amount", "paymode", "gpay", "cashier", "tender", "receiver", "grand", "tip", "thank"]):
                in_item_section = False
                print(f"End of item section: {line}")
                continue
            if not in_item_section:
                print(f"Skipped: {line} (outside item section)")
                continue
            # Get the next line (if available) to check for price/quantity
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # Debug the next-line check
            price_match = re.match(r"^(?:\$\s*)?\d+(?:[,.]\d{1,2})?(?:\s*\.?\s*)$", next_line)  # e.g., $9.50, 200.00, 200 00
            qty_match = re.match(r"^\d+(?:\.\d+)?(?:\s*[a-zA-Z]*)?$", next_line)  # e.g., 1, 1.000, 1 00o
            print(f"Next line check for '{line}': next_line='{next_line}', price_match={price_match}, qty_match={qty_match}")
            # Filter product names
            is_product = (
                len(line) > 2  # Short enough to be a product name
                and len(line) <= 50  # Avoid long strings
                and not any(keyword in line.lower() for keyword in [
                    "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp", "sri", "golden", "cholamb", "thanks", "cash", "order", "available", "id", "no:", "main", "pate", "edu", "rumullaivoyal", "thi", "for comming", "paymode", "product", "mrp", "amt", "off", "weight", "nateso", "atxpms", "bi17", "aty", "description", "tender", "receiver", "grand", "foods", "pm", "date", "time", "server", "type", "seq", "exp", "auth", "lane", "visa", "debit", "purchase", "saving"
                ])
                and not re.match(r"^\d+(\.\d+)?$", line)  # Exclude pure numbers/amounts
                and not re.match(r"^\d+$", line)  # Exclude pure integers
                and re.match(r"^[A-Za-z\s][A-Za-z0-9\s%#*+]*$", line)  # Starts with letter, allows numbers and some symbols
                and (price_match or qty_match)  # Followed by price/quantity
            )
            if is_product:
                product_names.append(line)
                print(f"Accepted: {line} (followed by: {next_line})")
            else:
                reason = (
                    "too short" if len(line) <= 2 else
                    "too long" if len(line) > 50 else
                    "contains excluded keyword" if any(keyword in line.lower() for keyword in [
                        "bill", "total", "amount", "gpay", "cashier", "qty", "rate", "net", "round", "items", "super", "market", "road", "ph:", "whatsapp", "sri", "golden", "cholamb", "thanks", "cash", "order", "available", "id", "no:", "main", "pate", "edu", "rumullaivoyal", "thi", "for comming", "paymode", "product", "mrp", "amt", "off", "weight", "nateso", "atxpms", "bi17", "aty", "description", "tender", "receiver", "grand", "foods", "pm", "date", "time", "server", "type", "seq", "exp", "auth", "lane", "visa", "debit", "purchase", "saving"
                    ]) else
                    "pure number/amount" if re.match(r"^\d+(\.\d+)?$", line) or re.match(r"^\d+$", line) else
                    "invalid format" if not re.match(r"^[A-Za-z\s][A-Za-z0-9\s%#*+]*$", line) else
                    "not followed by price/quantity" if not (price_match or qty_match) else
                    "unknown"
                )
                print(f"Rejected: {line} (reason: {reason})")

        # Create description from product names
        description = ', '.join(product_names[:5]) if product_names else "No products detected"
        print(f"Final Product Names: {product_names}")
        # print(f"AAmount: {total_amount}")

        return {
            "status": "success",
            "amount": total_amount,
            "items": items,  # From LayoutLM
            "description": description,  # From full_text
            "full_text": text,
            "totals": layoutlm_result.get("totals", []),
            "error": layoutlm_result.get("error", None)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Define the path to the receipt images directory
# image_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/test.png"
# image_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/finance/bills/toys.jpg"


# # Process each image

# result = extract_receipt_data_easyocr(image_path)
# print("Extraction Result:")
# print(f"Status: {result['status']}")
# if result["status"] == "success":
#     print(f"Total Amount: ${result['amount']:.2f}")
#     print(f"Items (LayoutLM): {result['items']}")
#     print(f"Totals (LayoutLM): {result['totals']}")
#     print(f"Description (full_text): {result['description']}")
#     print(f"Full Text:\n{result['full_text']}")
# else:
#     print(f"Error: {result['message']}")
# if result.get("error"):
#     print(f"LayoutLM Error: {result['error']}")