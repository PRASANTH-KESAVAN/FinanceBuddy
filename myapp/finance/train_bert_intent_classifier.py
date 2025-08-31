import pandas as pd
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import EvalPrediction
import torch
import numpy as np
import os
import json
from datetime import datetime
import nltk

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    raise

# Set project directory
BASE_DIR = r"C:\Prasanth\Prasanth\AllLanguages\PROJECT\FINANCE-ASSISTANCCE\myapp"
os.chdir(BASE_DIR)

# Logging setup
import logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Step 1: Analyze and Merge Datasets
def merge_datasets():
    try:
        # Load primary dataset
        data = pd.read_csv('./finance/intents1.csv')
        
        combined_data = data

        combined_data.to_csv('intents_combined.csv', index=False)
        logger.info(f"Saved intents_combined.csv with {len(combined_data)} records")

        # Analyze intent distribution
        intent_counts = combined_data['intent'].value_counts()
        print("Intent Distribution:")
        print(intent_counts)
        print(f"Total intents: {len(intent_counts)}")
        print(f"Intents with <50 examples: {sum(intent_counts < 50)}")
        logger.info(f"Intents with <50 examples: {sum(intent_counts < 50)}")

        return combined_data
    except Exception as e:
        logger.error(f"Error in merge_datasets: {e}")
        raise

# Step 2: Augment Dataset for Underrepresented Intents
def augment_dataset(data):
    try:
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_data = []
        intent_counts = data['intent'].value_counts()

        for intent, count in intent_counts.items():
            if count < 100:
                intent_data = data[data['intent'] == intent]
                logger.info(f"Augmenting intent {intent} with {count} examples")
                n_augment = max(3, (100 - count) // count)  # Ensure ~100 examples
                for text in intent_data['text']:
                    augmented_texts = aug.augment(text, n=n_augment)
                    for aug_text in augmented_texts:
                        augmented_data.append({'text': aug_text, 'intent': intent})

        # Manual additions for critical intents
        manual_additions = [
            {'text': 'What’s my stock profit this month?', 'intent': 'investment_get_profit'},
            {'text': 'Show my investment gains', 'intent': 'investment_get_profit'},
            {'text': 'Add ₹1000 electricity bill due tomorrow', 'intent': 'bill_add'},
            {'text': 'Log a ₹750 utility bill', 'intent': 'bill_add'},
            {'text': 'Track ₹300 for groceries', 'intent': 'expense_add'},
            {'text': 'Record ₹500 spent on dining', 'intent': 'expense_add'},
            {'text': 'What’s my food budget?', 'intent': 'budget_get'},
            {'text': 'Show my dining budget', 'intent': 'budget_get'},
            {'text': 'Update my name to Jane', 'intent': 'profile_update_name'},
            {'text': 'Change my profile name to Bob', 'intent': 'profile_update_name'},
            {'text': 'What’s my top investment?', 'intent': 'investment_get_max_profit'},
            {'text': 'Show my most profitable stock', 'intent': 'investment_get_max_profit'},
            {'text': 'What’s my user slug?', 'intent': 'profile_get_slug'},
            {'text': 'Show my account’s unique slug', 'intent': 'profile_get_slug'},
            {'text': 'Add a category for books', 'intent': 'category_add'},
            {'text': 'Create a pet expenses category', 'intent': 'category_add'},
            {'text': 'Add ₹2000 for salary income', 'intent': 'income_add'},
            {'text': 'Log ₹1500 freelance earnings', 'intent': 'income_add'},
            {'text': 'Invest ₹5000 in stocks', 'intent': 'investment_add'},
            {'text': 'Add ₹3000 to mutual funds', 'intent': 'investment_add'},
            {'text': 'Show my expense trends', 'intent': 'expense_get_trend'},
            {'text': 'What’s my spending pattern?', 'intent': 'expense_get_trend'},
            {'text': 'What’s my income trend?', 'intent': 'income_get_trend'},
            {'text': 'Show my earnings pattern', 'intent': 'income_get_trend'},
            {'text': 'What’s my investment trend?', 'intent': 'investment_get_trend'},
            {'text': 'Show my portfolio trends', 'intent': 'investment_get_trend'}
        ]
        augmented_data.extend(manual_additions)
        logger.info(f"Added {len(manual_additions)} manual examples")

        augmented_df = pd.DataFrame(augmented_data)
        combined_data = pd.concat([data, augmented_df]).drop_duplicates(subset=['text'])
        combined_data.to_csv('intents_augmented.csv', index=False)
        logger.info(f"Saved intents_augmented.csv with {len(combined_data)} records")

        return combined_data
    except Exception as e:
        logger.error(f"Error in augment_dataset: {e}")
        raise

# Step 3: Create Stratified Train-Test Split
def create_train_test_split(data):
    try:
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            stratify=data['intent'],
            random_state=42
        )
        train_data.to_csv('train.csv', index=False)
        test_data.to_csv('test.csv', index=False)
        logger.info(f"Created train.csv ({len(train_data)} records) and test.csv ({len(test_data)} records)")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error in create_train_test_split: {e}")
        raise

# Step 4: Prepare Dataset for Training
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 5: Compute Metrics for Evaluation
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Step 6: Train the Model
def train_model(train_data, test_data):
    try:
        # Initialize tokenizer and model
        model_name = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        intent_to_idx = {intent: idx for idx, intent in enumerate(sorted(train_data['intent'].unique()))}
        idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}

        # Prepare datasets
        train_dataset = IntentDataset(
            texts=train_data['text'].tolist(),
            labels=[intent_to_idx[intent] for intent in train_data['intent']],
            tokenizer=tokenizer
        )
        eval_dataset = IntentDataset(
            texts=test_data['text'].tolist(),
            labels=[intent_to_idx[intent] for intent in test_data['intent']],
            tokenizer=tokenizer
        )

        # Initialize model
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(intent_to_idx)
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy='epoch',  # Corrected from evaluation_strategy
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1'
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        # Train model
        logger.info("Starting model training")
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained('./results')
        tokenizer.save_pretrained('./results')
        logger.info("Model and tokenizer saved to ./results")

        # Save intent mappings
        with open('intent_mappings.json', 'w') as f:
            json.dump({'intent_to_idx': intent_to_idx, 'idx_to_intent': idx_to_intent}, f)
        logger.info("Intent mappings saved to intent_mappings.json")

        # Evaluate final model
        eval_results = trainer.evaluate()
        print("Final Evaluation Results:")
        print(eval_results)
        logger.info(f"Final evaluation results: {eval_results}")

        return eval_results
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise

# Step 7: Validate Model with Sample Queries
def validate_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('./results')
        model = DistilBertForSequenceClassification.from_pretrained('./results')
        model.eval()
        with open('intent_mappings.json', 'r') as f:
            mappings = json.load(f)
        idx_to_intent = mappings['idx_to_intent']

        test_queries = [
            "What’s my profile name?",
            "Add ₹500 electricity bill due next week",
            "Show my stock investment profit",
            "What’s my budget for food?",
            "Track ₹200 for travel",
            "Update my name to Alice",
            "What’s my most profitable investment?",
            "Show my unpaid bills",
            "Add ₹3000 for salary",
            "What are my expense categories?",
            "What’s my investment trend this year?",
            "What’s the weather today?"
        ]

        print("\nValidating model with sample queries:")
        for query in test_queries:
            inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = torch.argmax(logits, dim=-1).item()
            predicted_intent = idx_to_intent[str(predicted_idx)]
            confidence = torch.softmax(logits, dim=-1)[0, predicted_idx].item()
            print(f"Query: {query} | Intent: {predicted_intent} | Confidence: {confidence:.3f}")
            logger.info(f"Validation - Query: {query} | Intent: {predicted_intent} | Confidence: {confidence:.3f}")
    except Exception as e:
        logger.error(f"Error in validate_model: {e}")
        raise

# Main Execution
def main():
    try:
        print(f"[{datetime.now()}] Starting intent classifier training pipeline")
        logger.info("Starting intent classifier training pipeline")

        # Merge datasets
        combined_data = merge_datasets()

        # Augment dataset
        augmented_data = augment_dataset(combined_data)

        # Create train-test split
        train_data, test_data = create_train_test_split(augmented_data)

        # Train model
        eval_results = train_model(train_data, test_data)

        # Validate model
        validate_model()

        print(f"[{datetime.now()}] Training pipeline completed")
        logger.info(f"Training pipeline completed with F1 score: {eval_results.get('eval_f1', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()