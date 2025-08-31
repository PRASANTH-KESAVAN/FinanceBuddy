import pandas as pd
data = pd.read_csv('C:\Prasanth\Prasanth\AllLanguages\PROJECT\FINANCE-ASSISTANCCE\myapp\intents_augmented.csv')
intent_counts = data['intent'].value_counts()
print("Intent Distribution in intents_augmented.csv:")
print(intent_counts)
print(f"Total intents: {len(intent_counts)}")
print(f"Total records: {len(data)}")