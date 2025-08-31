# nlp_utils.py
import spacy

nlp = spacy.load("en_core_web_sm")

# def detect_intent(text):
#     doc = nlp(text.lower())

#     # Keyword mapping for basic intent classification
#     text = text.lower()

#     if "name" in text:
#         return "get_name"
#     elif "profile picture" in text or "image" in text or "photo" in text:
#         return "get_profile_pic"
#     elif "income" in text and "savings" in text:
#         return "get_income_savings"
#     elif "total income" in text or ("income" in text and "month" in text):
#         return "get_total_income"
#     elif "income sources" in text or "where" in text and "income" in text:
#         return "get_income_sources"
#     elif "spent" in text and "food" in text:
#         return "get_food_expense"
#     elif "spent" in text and "month" in text:
#         return "get_total_spent"
#     elif "expenses" in text and "last week" in text:
#         return "get_weekly_expenses"
#     elif "unpaid bills" in text:
#         return "get_unpaid_bills"
#     elif "upcoming bills" in text:
#         return "get_upcoming_bills"
#     elif "electricity bill" in text:
#         return "get_electricity_due"
#     elif "investments" in text and ("list" in text or "show" in text):
#         return "get_investments"
#     elif "profit" in text or "loss" in text:
#         return "get_profit_loss"
#     else:
#         return "unknown"


def detect_intent(text):
    doc = nlp(text.lower())

    # Keyword mapping for basic intent classification
    text = text.lower()

    if "name" in text:
        return "get_name"
    elif "profile picture" in text or "image" in text or "photo" in text:
        return "get_profile_pic"
    elif "income" in text and "savings" in text:
        return "get_income_savings"
    elif "total income" in text or ("income" in text and "month" in text):
        return "get_total_income"
    elif "income sources" in text or "where" in text and "income" in text:
        return "get_income_sources"
    elif "spent" in text and "food" in text:
        return "get_food_expense"
    elif "spent" in text and "month" in text:
        return "get_total_spent"
    elif "expenses" in text and "last week" in text:
        return "get_weekly_expenses"
    elif "unpaid bills" in text:
        return "get_unpaid_bills"
    elif "upcoming bills" in text:
        return "get_upcoming_bills"
    elif "electricity bill" in text:
        return "get_electricity_due"
    elif "investments" in text and ("list" in text or "show" in text or "my" in text):
        return "get_investments"
    elif "profit" in text or "loss" in text:
        return "get_profit_loss"
    elif "expenses" in text and "category" in text:
        return "get_expense_categories"
    elif "total expenses" in text:
        return "get_total_expenses"
    else:
        return "unknown"





# # nlp_utils.py
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import pickle
# import logging

# logger = logging.getLogger(__name__)

# try:
#     tokenizer = BertTokenizer.from_pretrained('./bert_intent_model')
#     model = BertForSequenceClassification.from_pretrained('./bert_intent_model')
#     with open('label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)
#     logger.info("BERT model and tokenizer loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load BERT model: {e}")
#     raise

# def detect_intent(text):
#     try:
#         text = text.lower()
#         inputs = tokenizer(
#             text,
#             return_tensors='pt',
#             max_length=128,
#             padding='max_length',
#             truncation=True
#         )
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             predicted_class = torch.argmax(logits, dim=1).item()
#         intent = label_encoder.inverse_transform([predicted_class])[0]
#         logger.debug(f"Query: {text} -> Intent: {intent}")
#         return intent
#     except Exception as e:
#         logger.error(f"Error detecting intent: {e}")
#         return "unknown"


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