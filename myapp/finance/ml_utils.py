# 

from .models import Investment, InvestmentCategory
import numpy as np
import re

import pandas as pd

def analyze_investments(user, investment_type=None):
    investments = Investment.objects.filter(user=user)
    if investment_type:
        investments = investments.filter(investment_type__name=investment_type)

    if not investments.exists():
        return {"status": "No investments found."}

    # Calculate ROI
    investment_data = []
    for inv in investments:
        # Convert Decimal values to float to avoid TypeError
        roi = (float(inv.profit) - float(inv.loss)) / float(inv.amount)
        investment_data.append({
            "type": inv.investment_type.name,
            "roi": roi
        })

    df = pd.DataFrame(investment_data)
    avg_roi = df.groupby("type")["roi"].mean().sort_values(ascending=False)
    user_latest = df.iloc[-1]
    user_type = user_latest["type"]
    user_roi = user_latest["roi"]

    recommendation = None
    for alt_type, alt_roi in avg_roi.items():
        if alt_type != user_type and alt_roi > user_roi + 0.05:
            recommendation = f"Your {user_type} investment is underperforming. Consider switching to {alt_type}."
            break

    return {
        "user_investment": user_type,
        "user_roi": round(user_roi * 100, 2),
        "avg_rois": avg_roi.to_dict(),
        "recommendation": recommendation or "Your investments are performing well."
    }



import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
from .models import Investment

def forecast_roi(user, months_ahead=6, investment_type=None):
    investments = Investment.objects.filter(user=user).order_by('date')
    if investment_type:
        investments = investments.filter(investment_type__name=investment_type)

    if not investments.exists():
        return {"status": "No investment data found for forecasting."}

    data = []
    for inv in investments:
        date_obj = inv.date
        # Convert Decimal values to float to avoid TypeError
        roi = (float(inv.profit) - float(inv.loss)) / float(inv.amount)
        data.append({'date': date_obj, 'roi': roi})

    df = pd.DataFrame(data)
    df['timestamp'] = df['date'].map(lambda x: x.toordinal())

    model = LinearRegression()
    model.fit(df[['timestamp']], df['roi'])

    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date, periods=months_ahead + 1, freq='M')[1:]
    future_timestamps = future_dates.map(lambda d: d.toordinal()).values.reshape(-1, 1)
    predictions = model.predict(future_timestamps)

    results = list(zip(future_dates.strftime('%b %Y'), (predictions * 100).round(2)))

    return {
        "status": "Forecast successful.",
        "forecast": results
    }



# ml_utils.py

import pytesseract
from PIL import Image
import joblib
import re
# import easyocr

# Load ML category model and vectorizer
model = joblib.load("category_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

'''
first worked method
'''
# ---------- SMART RECEIPT OCR FUNCTIONS (Based on Receipt-Scanner repo) -----------

def extract_text_from_image(image_path):
    """Simple OCR extraction."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

import easyocr
import cv2
import re
from PIL import Image

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

def extract_receipt_data(image_path):
    # """Smart OCR: Extract important receipt fields."""
    # img = Image.open(image_path)
    # text = pytesseract.image_to_string(img)

    # # Extract amounts using regex
    # amounts = re.findall(r"\d+\.\d{2}", text)
    # if amounts:
    #     amounts = [float(a) for a in amounts]
    #     total_amount = max(amounts)  # Take highest value (usually total)
    # else:
    #     total_amount = 0.0

    # # Take first few lines as description
    # lines = text.strip().split('\n')
    # description_lines = []
    # for line in lines:
    #     line = line.strip()
    #     if len(line) > 3:
    #         description_lines.append(line)
    #     if len(description_lines) >= 5:
    #         break

    # description = ', '.join(description_lines)

    # return {
    #     'amount': total_amount,
    #     'description': description,
    #     'full_text': text
    # }
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": "Failed to load image"}

    # Run EasyOCR
    results = reader.readtext(img, detail=0)
    text = "\n".join(results)

    # Extract amounts
    amounts = re.findall(r"\d+\.\d{2}", text)
    total_amount = max([float(a) for a in amounts]) if amounts else 0.0

    # Extract items with improved regex
    # Matches lines like "Item Name  1  12.34" or "Item Name 12.34"
    items = []

    # Create description from items
    description = ', '.join(items[:5]) if items else "No items detected"

    return {
        'amount': total_amount,
        'items': items,
        'description': description,
        'full_text': text
    }  





from .models import Expense, Income, Category, IncomeCategory
from datetime import date

# def update_investment_effect(user, investment, action="add"):
#     # Create/Get Investment-related Categories
#     investment_category, _ = Category.objects.get_or_create(name="Investment")
#     investment_income_category, _ = IncomeCategory.objects.get_or_create(name="Investment Returns")

#     if action == "add":
#         # Log the investment as an expense
#         Expense.objects.create(
#             user=user,
#             type="Investment",
#             category=investment_category,
#             amount=int(investment.amount),
#             date=investment.date,
#             notes=f"Invested in {investment.investment_type.name}"
#         )

#         # Handle profit or loss
#         net_result = investment.profit - investment.loss
#         if net_result != 0:
#             Income.objects.create(
#                 user=user,
#                 category=investment_income_category,
#                 amount=int(net_result),
#                 date=investment.date,
#                 notes=f"Result of investment in {investment.investment_type.name}"
#             )

#     elif action == "delete":
#         # Delete related expense and income by notes (simplified logic)
#         Expense.objects.filter(
#             user=user, type="Investment",
#             amount=int(investment.amount),
#             date=investment.date,
#             notes__icontains=investment.investment_type.name
#         ).delete()

#         Income.objects.filter(
#             user=user,
#             category=investment_income_category,
#             date=investment.date,
#             notes__icontains=investment.investment_type.name
#         ).delete()

from .models import Expense, Category, Profile
from datetime import date

def update_investment_effect(user, investment, action="add"):
    investment_category, _ = Category.objects.get_or_create(name="Investment")
    
    # Ensure profile exists
    profile = Profile.objects.get(user=user)

    if action == "add":
        # Add expense only once during investment month
        Expense.objects.create(
            user=user,
            type="Investment",
            category=investment_category,
            amount=investment.amount,
            date=investment.date,
            notes="Investment added"
        )
        
        # Update savings considering profit and loss
        profile.savings += investment.profit
        profile.savings -= investment.loss
        profile.save()

    elif action == "delete":
        # Remove matching expense (by date and amount)
        Expense.objects.filter(
            user=user,
            category=investment_category,
            amount=investment.amount,
            date=investment.date
        ).delete()

        # Reverse profit and loss effect
        profile.savings -= investment.profit
        profile.savings += investment.loss
        profile.save()








# import cv2
# import numpy as np
# from paddleocr import PaddleOCR
# import paddle

# paddle.set_device('cpu')


# # Initialize PaddleOCR model once
# # ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
# ocr_model = PaddleOCR(use_angle_cls=True, lang='en', device='cpu')  # For CPU usage


# def preprocess_image(image_path):
#     """Enhance image quality before OCR"""
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
#     _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     coords = np.column_stack(np.where(thresh > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     (h, w) = thresh.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     deskewed = cv2.warpAffine(thresh, M, (w, h),
#                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return deskewed

# import cv2

# def extract_text_deep(image_path):
#     """Use PaddleOCR properly on normal images"""
#     img = cv2.imread(image_path)  # Read original color image

#     if img is None:
#         return ""

#     # Give raw color image to PaddleOCR (very important)
#     result = ocr_model.ocr(img, cls=True)

#     if not result or not result[0]:
#         return ""

#     extracted_text = ""
#     for line in result[0]:
#         extracted_text += line[1][0] + "\n"

#     return extracted_text




# # def extract_entities(text):
# #     """Extract total and items from text"""
# #     import re
# #     total_pattern = r"Total\s*[:\-]?\s*\$?(\d+\.\d{2})"
# #     total_match = re.search(total_pattern, text, re.IGNORECASE)
# #     total_amount = float(total_match.group(1)) if total_match else 0.0

# #     item_pattern = r"(.+?)\s+\d+\s+\d+\.\d{2}"
# #     items = re.findall(item_pattern, text)

# #     return total_amount, items

# def extract_entities(raw_text):
#     # Find total amount
#     total_pattern = r"Total\s*[:\-]?\s*\$?(\d+\.\d{2})"
#     total_match = re.search(total_pattern, raw_text, re.IGNORECASE)
#     total_amount = float(total_match.group(1)) if total_match else None

#     # Extract item lines (basic rule: lines with "x qty price")
#     item_pattern = r"(.+?)\s+\d+\s+\d+\.\d{2}"
#     items = re.findall(item_pattern, raw_text)

#     # return {
#     #     "total_amount": total_amount,
#     #     "items": items
#     # }
    
#     return total_amount, items


# def extract_receipt_data_advanced(image_path):
#     """Receipt extraction pipeline without overprocessing"""
#     # No preprocessing - work on original uploaded image

#     # Step 1: OCR
#     text = extract_text_deep(image_path)

#     # Step 2: NLP Entity Extraction
#     total_amount, items = extract_entities(text)

#     # Step 3: Fallback if needed
#     if total_amount is None:
#         amounts = re.findall(r"\d+\.\d{2}", text)
#         if amounts:
#             amounts = [float(a) for a in amounts]
#             total_amount = max(amounts)

#     # Step 4: Description
#     description_lines = text.strip().split('\n')[:5]
#     description = ', '.join(description_lines)

#     return {
#         'amount': total_amount,
#         'items': items,
#         'description': description,
#         'full_text': text
#     }




# def predict_category(text):
#     """Predict category from receipt text using pipeline."""
#     clean_text = re.sub(r'[^a-zA-Z\s]', '', text)  # simple cleaning
#     clean_text = clean_text.lower().strip()
#     prediction = model.predict([clean_text])  # Pass as list of one document
#     return prediction[0]





'''
it worked for the before
'''


# def predict_category(text):
#     """Predict category from receipt text using pipeline."""
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # basic clean-up
#     prediction = model.predict([text])  # <-- notice passing [text] (as list)
#     return prediction[0]

# ml_utils.py
import re
import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

# Load model and tokenizer
model_path = "C:/Prasanth/Prasanth/AllLanguages/PROJECT/FINANCE-ASSISTANCCE/myapp/layoutlm-finetuned-sroie"
try:
    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
    model = LayoutLMForTokenClassification.from_pretrained(model_path)
except Exception as e:
    raise Exception(f"Error loading model or tokenizer from {model_path}: {str(e)}")

# Define available categories
AVAILABLE_CATEGORIES = [
    "Food", "Entertainment", "Rent", "Groceries", "Utilities",
    "Transportation", "Healthcare", "Education", "Insurance",
    "Gifts & Donations", "Dining Out", "Subscriptions",
    "Home Maintenance", "Personal Care", "Childcare", "Debt Payments",
    "Cloths & Accessories"
]

# Keyword mapping for category prediction
CATEGORY_KEYWORDS = {
    "Food": ["chips", "snack", "tomato", "cake", "bread", "fruit", "vegetable", "meat", "dairy"],
    "Groceries": ["grocery", "food", "beverage", "milk", "egg", "cereal", "pasta", "rice"],
    "Dining Out": ["restaurant", "cafe", "dinner", "lunch", "breakfast", "coffee"],
    "Entertainment": ["movie", "ticket", "concert", "game", "event"],
    "Rent": ["rent", "apartment", "housing"],
    "Utilities": ["electricity", "water", "gas", "internet", "phone", "bill"],
    "Transportation": ["fuel", "gasoline", "bus", "train", "taxi", "uber", "parking"],
    "Healthcare": ["medicine", "pharmacy", "doctor", "hospital", "health"],
    "Education": ["book", "tuition", "school", "course", "training"],
    "Insurance": ["insurance", "premium", "policy"],
    "Gifts & Donations": ["gift", "donation", "charity", "present"],
    "Subscriptions": ["subscription", "netflix", "spotify", "magazine", "membership"],
    "Home Maintenance": ["repair", "plumbing", "cleaning", "hardware"],
    "Personal Care": ["cosmetic", "shampoo", "soap", "haircut", "grooming"],
    "Childcare": ["diaper", "babysitter", "daycare", "toy"],
    "Debt Payments": ["loan", "credit", "payment", "mortgage"],
    "Cloths & Accessories": ["cloth", "shoe", "shirt", "dress", "jacket", "accessory"]
}

def predict_category(text, device='cpu'):
    """Predict category from receipt text using LayoutLM and keyword mapping."""
    if not text or not isinstance(text, str):
        return "Unknown"

    # Basic clean-up
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip().lower()
    if not text:
        return "Unknown"

    # Step 1: Try LayoutLM to identify if the text is an item
    is_item = True  # Default to True for robustness
    try:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        encoding["bbox"] = torch.zeros((1, encoding["input_ids"].shape[1], 4), dtype=torch.long)
        encoding = {key: val.to(device) for key, val in encoding.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)

        predicted_labels = []
        for token_id, pred in zip(encoding["input_ids"][0], predictions[0]):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                continue
            label = model.config.id2label[pred.item()]
            predicted_labels.append(label)

        print(f"LayoutLM Predicted Labels for '{text}': {predicted_labels}")
        is_item = any(label in ["B-ITEM", "I-ITEM"] for label in predicted_labels)
    except Exception as e:
        print(f"Error in LayoutLM inference: {str(e)}")
        is_item = True  # Fallback: assume it's an item

    # Step 2: Map to a category if it's an item
    if is_item:
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                print(f"Matched category '{category}' for text '{text}'")
                return category
        # Default to "Groceries" for items not matching specific keywords
        print(f"Defaulting to 'Groceries' for text '{text}'")
        return "Groceries"
    
    # Step 3: Handle non-item cases
    print(f"Text '{text}' classified as non-item, returning 'Unknown'")
    return "Unknown"



###################    Phase 2         ########################################
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import logging

# Set up logging
logger = logging.getLogger(__name__)

# def prepare_lstm_data(expense_queryset, sequence_length=10):
#     df = pd.DataFrame.from_records(expense_queryset.values('date', 'amount'))
#     if df.empty:
#         logger.warning("Expense data is empty in prepare_lstm_data")
#         return None, None, None, None

#     # Rename columns
#     df.columns = ['ds', 'y']
#     df['ds'] = pd.to_datetime(df['ds'])

#     # Group by date and sum amounts
#     df = df.groupby('ds')['y'].sum().reset_index()
#     df = df.sort_values('ds')

#     # Create a complete date range to handle missing dates
#     start_date = df['ds'].min()
#     end_date = df['ds'].max()
#     all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
#     df_full = pd.DataFrame(all_dates, columns=['ds'])
#     df_full = df_full.merge(df, on='ds', how='left').fillna({'y': 0})

#     # Scale the amounts
#     scaler = MinMaxScaler()
#     df_full['y_scaled'] = scaler.fit_transform(df_full[['y']])

#     # Adjust sequence length dynamically
#     sequence_length = min(sequence_length, len(df_full) - 1)
#     if sequence_length < 2:
#         logger.warning(f"Sequence length too small: {sequence_length}, data points: {len(df_full)}")
#         return None, None, None, None

#     # Prepare sequences
#     sequence_data, target_data = [], []
#     for i in range(len(df_full) - sequence_length):
#         sequence_data.append(df_full['y_scaled'].values[i:i + sequence_length])
#         target_data.append(df_full['y_scaled'].values[i + sequence_length])

#     if not sequence_data:
#         logger.warning(f"No sequences generated: {len(df_full)} records, sequence_length={sequence_length}")
#         return None, None, None, None

#     X = np.array(sequence_data).reshape(-1, sequence_length, 1)
#     y = np.array(target_data)

#     logger.info(f"Prepared LSTM data: X_shape={X.shape}, y_shape={y.shape}, sequence_length={sequence_length}")
#     return X, y, scaler, df_full

# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# from tensorflow.keras.optimizers import Adam
# import logging
# from datetime import timedelta

# # Set up logging
# logger = logging.getLogger(__name__)

# def prepare_lstm_data(expense_queryset, sequence_length=30):
#     df = pd.DataFrame.from_records(expense_queryset.values('date', 'amount'))
#     if df.empty:
#         logger.warning("Expense data is empty in prepare_lstm_data")
#         return None, None, None, None

#     df.columns = ['ds', 'y']
#     df['ds'] = pd.to_datetime(df['ds'])
#     df = df.groupby('ds')['y'].sum().reset_index().sort_values('ds')

#     # Handle outliers
#     q1, q3 = df['y'].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     df = df[(df['y'] >= q1 - 1.5 * iqr) & (df['y'] <= q3 + 1.5 * iqr)]

#     # Create complete date range
#     start_date = df['ds'].min()
#     end_date = df['ds'].max()
#     all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
#     df_full = pd.DataFrame(all_dates, columns=['ds'])
#     df_full = df_full.merge(df, on='ds', how='left')
#     df_full['y'] = df_full['y'].ffill().bfill()  # Updated to use ffill() and bfill()

#     # Add features
#     df_full['day_of_week'] = df_full['ds'].dt.dayofweek
#     df_full['month'] = df_full['ds'].dt.month

#     # Scale data
#     scaler = MinMaxScaler()
#     df_full['y_scaled'] = scaler.fit_transform(df_full[['y']])
#     scaler_features = MinMaxScaler()
#     df_full[['day_of_week', 'month']] = scaler_features.fit_transform(df_full[['day_of_week', 'month']])

#     sequence_length = min(sequence_length, len(df_full) - 1)
#     if sequence_length < 2:
#         logger.warning(f"Sequence length too small: {sequence_length}, data points: {len(df_full)}")
#         return None, None, None, None

#     sequence_data, target_data = [], []
#     for i in range(len(df_full) - sequence_length):
#         sequence_data.append(df_full[['y_scaled', 'day_of_week', 'month']].values[i:i + sequence_length])
#         target_data.append(df_full['y_scaled'].values[i + sequence_length])

#     if not sequence_data:
#         logger.warning(f"No sequences generated: {len(df_full)} records, sequence_length={sequence_length}")
#         return None, None, None, None

#     X = np.array(sequence_data)
#     y = np.array(target_data)
#     logger.info(f"Prepared LSTM data: X_shape={X.shape}, y_shape={y.shape}, sequence_length={sequence_length}")
#     return X, y, scaler, df_full, scaler_features  # Return scaler_features for use in forecasting

# def build_lstm_model(input_shape):
#     model = Sequential([
#         Input(shape=input_shape),
#         LSTM(100, activation='tanh', return_sequences=True),
#         Dropout(0.2),
#         LSTM(50, activation='tanh', return_sequences=False),
#         Dropout(0.2),
#         Dense(25, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model

# def forecast_expense_lstm(model, last_sequence, future_steps, scaler, df_full, scaler_features, sequence_length):
#     predictions = []
#     input_seq = last_sequence.reshape(1, sequence_length, -1)  # Shape: (1, sequence_length, 3)

#     # Get the last date from df_full
#     last_date = df_full['ds'].max()

#     # Create future dates
#     future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_steps, freq='D')
#     future_df = pd.DataFrame(future_dates, columns=['ds'])
#     future_df['day_of_week'] = future_df['ds'].dt.dayofweek
#     future_df['month'] = future_df['ds'].dt.month
#     future_df[['day_of_week', 'month']] = scaler_features.transform(future_df[['day_of_week', 'month']])

#     for i in range(future_steps):
#         # Predict the next value
#         next_pred = model.predict(input_seq, verbose=0)[0][0]
#         predictions.append(next_pred)

#         # Prepare the next input sequence
#         next_day_features = future_df.iloc[i][['day_of_week', 'month']].values
#         next_input = np.array([next_pred, *next_day_features]).reshape(1, 1, 3)  # Shape: (1, 1, 3)
#         input_seq = np.append(input_seq[:, 1:, :], next_input, axis=1)  # Shape: (1, sequence_length, 3)

#     # Inverse transform predictions
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
#     return predictions





########################    ANOMALY DETECTION           ##################################

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import joblib
import os
from .models import Expense

logger = logging.getLogger(__name__)

def get_user_expenses(user, start_date=None):
    """Fetch user expenses, optionally filtered by start date."""
    queryset = Expense.objects.filter(user=user)
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    return [
        {
            "amount": exp.amount,
            "category": exp.category.name,
            "date": exp.date,
            "type": exp.type,
            "notes": exp.notes
        }
        for exp in queryset
    ]

def preprocess_expense_data(user, start_date=None):
    """Preprocess expense data with enhanced feature engineering."""
    data = get_user_expenses(user, start_date)
    df = pd.DataFrame(data)

    if df.empty:
        logger.warning(f"Empty expense data for user {user.id}")
        return df

    # Validate data
    if df["amount"].lt(0).any():
        logger.warning(f"Negative amounts found for user {user.id}")
        df = df[df["amount"] >= 0]

    # Convert and extract date features
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_day"] = df["date"].dt.weekday
    df["days_since_last"] = df.groupby("category")["date"].diff().dt.days.fillna(0)

    # Category-based features
    df["category_count"] = df.groupby("category")["category"].transform("count")
    df["category_avg_amount"] = df.groupby("category")["amount"].transform("mean")

    # Normalize amount
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["amount"]])

    # Optionally encode categorical variables
    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    logger.info(f"Preprocessed data for user {user.id}: {df.shape}, columns: {df.columns}")
    return df, scaler

def train_anomaly_detection_model_ml(user, start_date=None, save_model=True):
    """Train anomaly detection model with enhanced features and persistence."""
    df, scaler = preprocess_expense_data(user, start_date)

    if df.empty or len(df) < 5:
        logger.warning(f"Insufficient data for user {user.id}: {len(df)} records")
        return pd.DataFrame(columns=["amount", "category", "date", "month", "day", "week_day", "category_count", "scaled_amount", "anomaly"])

    # Dynamic contamination based on data size
    contamination = min(0.1, 5 / len(df)) if len(df) > 50 else 0.05

    # Select features (including new ones and encoded categories)
    feature_cols = [
        "scaled_amount", "month", "day", "week_day", "category_count",
        "days_since_last", "category_avg_amount"
    ] + [col for col in df.columns if col.startswith("cat_")]

    features = df[feature_cols]
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(features)

    df["anomaly"] = model.predict(features)
    df["anomaly"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    # Save model and scaler
    if save_model:
        model_dir = f"models/user_{user.id}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/anomaly_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    logger.info(f"Anomaly detection trained for user {user.id}: {features.describe().to_dict()}")
    return df


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .models import Expense, Category
import logging

logger = logging.getLogger(__name__)

def prepare_clustering_data(user=None, by_month=False, start_date=None):
    """Prepare data for clustering spending patterns by user or month."""
    # Fetch expenses
    if user:
        expenses = Expense.objects.filter(user=user)
    else:
        expenses = Expense.objects.all()
    
    if start_date:
        expenses = expenses.filter(date__gte=start_date)

    if not expenses.exists():
        logger.warning("No expense data available for clustering")
        return None, None

    # Convert to DataFrame
    data = [
        {
            "user_id": exp.user.id,
            "category": exp.category.name,
            "amount": float(exp.amount),
            "date": exp.date,
        }
        for exp in expenses
    ]
    df = pd.DataFrame(data)

    # Feature engineering
    if by_month:
        # Aggregate by user and month
        df["month"] = df["date"].apply(lambda x: x.strftime("%Y-%m"))
        group_cols = ["user_id", "month"]
    else:
        # Aggregate by user
        group_cols = ["user_id"]

    # Pivot table for category-wise spending
    category_pivot = df.pivot_table(
        values="amount",
        index=group_cols,
        columns="category",
        aggfunc="sum",
        fill_value=0
    )

    # Additional features
    features = df.groupby(group_cols).agg({
        "amount": ["sum", "mean", "count"],
        "date": lambda x: (x.max() - x.min()).days if len(x) > 1 else 1
    }).reset_index()

    # Flatten column names
    features.columns = [
        "_".join(col).strip() if col[1] else col[0]
        for col in features.columns.values
    ]

    # Rename columns
    features = features.rename(columns={
        "amount_sum": "total_spending",
        "amount_mean": "avg_transaction",
        "amount_count": "transaction_count",
        "date_<lambda>": "active_days"
    })

    # Calculate spending frequency
    features["spending_frequency"] = features["transaction_count"] / features["active_days"].clip(lower=1)

    # Merge category-wise spending with other features
    features = features.merge(category_pivot, left_on=group_cols, right_index=True)

    # Handle missing categories
    all_categories = Category.objects.values_list("name", flat=True)
    for cat in all_categories:
        if cat not in features.columns:
            features[cat] = 0

    # Select feature columns
    feature_cols = ["total_spending", "avg_transaction", "transaction_count", "spending_frequency"] + list(all_categories)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=features.index)

    logger.info(f"Prepared clustering data: {scaled_df.shape}, features: {feature_cols}")
    return scaled_df, features[["user_id"] + (["month"] if by_month else [])]



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger(__name__)

def cluster_spending_patterns(data, max_clusters=10, random_state=42):
    """Apply K-Means clustering to spending data."""
    if data is None or data.empty:
        logger.warning("No data provided for clustering")
        return None, None, None

    n_samples = len(data)
    if n_samples < 2:
        logger.warning(f"Insufficient samples for clustering: {n_samples}. Need at least 2 samples.")
        return np.array([0] * n_samples), 1, None  # Assign single cluster for single sample

    # Determine optimal number of clusters using elbow method
    inertia = []
    silhouette_scores = []
    max_clusters = min(max_clusters, n_samples)  # Ensure max_clusters <= n_samples
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        if n_clusters > 1 and n_samples > n_clusters:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(0)

    # Choose optimal clusters (elbow point or max silhouette score)
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)] if silhouette_scores else 2

    # Ensure optimal_clusters <= n_samples
    optimal_clusters = min(optimal_clusters, n_samples)

    # Apply K-Means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(data)

    # Plot elbow curve (for debugging, optional)
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, inertia, marker="o")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig("elbow_plot.png")
    plt.close()

    logger.info(f"Clustered data into {optimal_clusters} clusters")
    return clusters, optimal_clusters, kmeans

from sklearn.cluster import DBSCAN

# def cluster_spending_patterns(data, eps=0.5, min_samples=5, random_state=42):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     clusters = dbscan.fit_predict(data)
#     n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
#     logger.info(f"Clustered data into {n_clusters} clusters with DBSCAN")
#     return clusters, n_clusters, dbscan


# from sklearn.mixture import GaussianMixture

# def cluster_spending_patterns(data, max_clusters=10, random_state=42):
#     bic = []
#     cluster_range = range(2, min(max_clusters + 1, len(data)))
#     for n_clusters in cluster_range:
#         gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
#         gmm.fit(data)
#         bic.append(gmm.bic(data))
#     optimal_clusters = cluster_range[np.argmin(bic)]
#     gmm = GaussianMixture(n_components=optimal_clusters, random_state=random_state)
#     clusters = gmm.fit_predict(data)
#     logger.info(f"Clustered data into {optimal_clusters} clusters with GMM")
#     return clusters, optimal_clusters, gmm



from sklearn.manifold import TSNE
import plotly.express as px
import plotly.offline as opy

def visualize_clusters(data, clusters, identifiers, title="Spending Patterns"):
    """Visualize clusters using t-SNE."""
    if data is None or clusters is None:
        logger.warning("No data or clusters for visualization")
        return None

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "Cluster": [f"Cluster {c}" for c in clusters]
    })
    plot_df = plot_df.merge(identifiers, left_index=True, right_index=True)

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["user_id"] + (["month"] if "month" in identifiers.columns else []),
        title=title,
        labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2"}
    )

    chart_html = opy.plot(fig, auto_open=False, output_type="div")
    return chart_html



########################### ocr  #########################################




import os
import re
import torch
import pytesseract
import easyocr
import cv2
import numpy as np
from PIL import Image
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from pathlib import Path

# Define labels (same as during training)
label_list = ["O", "B-ITEM", "I-ITEM", "B-TOTAL", "I-TOTAL"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Load trained model and tokenizer
model_path = "./layoutlm-finetuned-sroie"
tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
model = LayoutLMForTokenClassification.from_pretrained(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# LayoutLM inference function
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

    except Exception as e:
        return {"items": [], "totals": [], "error": str(e)}

# EasyOCR extraction function
def extract_receipt_data_easyocr(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Failed to load image"}

        # Run EasyOCR
        results = reader.readtext(img, detail=0)
        text = "\n".join(results)

        # Extract amounts (e.g., 12.34)
        amounts = re.findall(r"\d+\.\d{2}", text)
        total_amount = max([float(a) for a in amounts]) if amounts else 0.0

        # Get items from LayoutLM
        layoutlm_result = extract_entities_layoutlm(image_path)
        items = layoutlm_result.get("items", [])

        # Create description from items
        description = ', '.join(items[:5]) if items else "No items detected"

        return {
            "status": "success",
            "amount": total_amount,
            "items": items,
            "description": description,
            "full_text": text,
            "totals": layoutlm_result.get("totals", []),
            "error": layoutlm_result.get("error", None)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Test inference on a new receipt
# if __name__ == "__main__":
#     new_receipt_path = '/content/toys.jpg'  # Replace with your image path
#     if not os.path.exists(new_receipt_path):
#         print(f"Image not found at {new_receipt_path}")
#     else:
#         result = extract_receipt_data_easyocr(new_receipt_path)
#         print("Extraction Result:")
#         print(f"Status: {result['status']}")
#         if result["status"] == "success":
#             print(f"Total Amount: ${result['amount']:.2f}")
#             print(f"Items: {result['items']}")
#             print(f"Totals: {result['totals']}")
#             print(f"Description: {result['description']}")
#             print(f"Full Text:\n{result['full_text']}")
#         else:
#             print(f"Error: {result['message']}")
#         if result.get("error"):
#             print(f"LayoutLM Error: {result['error']}")









#     # return {
#     #     'amount': total_amount,
#     #     'description': description,
#     #     'full_text': text
    # }