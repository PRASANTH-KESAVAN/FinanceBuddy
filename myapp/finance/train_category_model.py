# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import joblib

# # Example data (you should improve with a better labeled dataset)
# data = {
#     "text": [
#         "Uber ride", "Netflix subscription", "Dominos pizza", "Tuition fees",
#         "Doctor appointment", "Electricity bill", "H&M shopping", "Haircut",
#         "Rent for house", "Diapers for baby", "Home cleaning"
#     ],
#     "category": [
#         "Transportation", "Subscriptions", "Food", "Education",
#         "Healthcare", "Utilities", "Cloths & Accessories", "Personal Care",
#         "Rent", "Childcare", "Home Maintenance"
#     ]
# }

# df = pd.DataFrame(data)

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df["text"])
# y = df["category"]

# model = MultinomialNB()
# model.fit(X, y)

# # Save the model and vectorizer
# joblib.dump(model, "category_classifier.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")



import pandas as pd

# Expanded example dataset
data = {
    "text": [
        "Uber ride", "Ola cab", "Bus fare", "Netflix subscription", "Amazon Prime", "Spotify plan",
        "Dominos pizza", "Lunch at McDonald's", "Grocery shopping", "Tuition fees", "School books",
        "Doctor appointment", "Hospital bill", "Electricity bill", "Gas connection", "Internet recharge",
        "Shopping at H&M", "Levi’s jeans", "Haircut", "Salon visit", "Rent for house", "Diapers for baby",
        "Babysitter fee", "House cleaning", "Plumbing repair"
    ],
    "category": [
        "Transportation", "Transportation", "Transportation", "Subscriptions", "Subscriptions", "Subscriptions",
        "Food", "Food", "Food", "Education", "Education",
        "Healthcare", "Healthcare", "Utilities", "Utilities", "Utilities",
        "Cloths & Accessories", "Cloths & Accessories", "Personal Care", "Personal Care",
        "Rent", "Childcare", "Childcare", "Home Maintenance", "Home Maintenance"
    ]
}
df = pd.DataFrame(data)

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and digits
    return text

df["text"] = df["text"].apply(clean_text)


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

params = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.1, 0.5, 1.0]
}

# grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1)

grid.fit(df["text"], df["category"])

print("Best Parameters:", grid.best_params_)


import joblib

joblib.dump(grid.best_estimator_, "category_classifier.pkl")
