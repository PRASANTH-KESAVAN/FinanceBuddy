# nlp_utils.py
import spacy

nlp = spacy.load("en_core_web_sm")


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import logging

logger = logging.getLogger(__name__)
model = DistilBertForSequenceClassification.from_pretrained('./results')
tokenizer = DistilBertTokenizer.from_pretrained('./results')
model.eval()

with open('intent_mappings.json', 'r') as f:
    mappings = json.load(f)
idx_to_intent = mappings['idx_to_intent']

def detect_intent(query):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_idx = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0, predicted_idx].item()
    intent = idx_to_intent[str(predicted_idx)]
    logger.info(f"Query: {query} | Intent: {intent} | Confidence: {confidence:.3f}")
    print(f"Query: {query} | Intent: {intent} | Confidence: {confidence:.3f}")
    return intent
