

# FinanceBuddy

FinanceBuddy is an AI-powered **personal finance assistant** built with Django.
It helps users **track expenses, manage budgets, set bill reminders, analyze income/investments, and interact with an NLP-based chatbot** for quick insights.

---

## 🚀 Features

* 💰 **Expense & Income Tracking** – Log daily transactions and categorize them
* 📊 **Budgeting & Reports** – Set budgets and visualize spending patterns
* 🧾 **Bill Management** – Upload bills (OCR support to auto-extract details)
* 🤖 **AI Chatbot** – Chat with a virtual finance assistant powered by NLP & BERT
* 📈 **Investment Tracking** – Manage savings & investments in one place
* 🔔 **Reminders** – Get alerts for upcoming bills & recurring transactions
* 🧠 **Machine Learning Models**

  * Intent classification (BERT-based)
  * Anomaly detection in expenses
  * Automated categorization of transactions
  * Speech Recognition for Expense Logging

---

## 🛠️ Tech Stack

* **Backend:** Django, Python
* **Frontend:** HTML, CSS (Tailwind), Django Templates
* **Database:** MySQL / PostgreSQL
* **AI/ML:** Scikit-learn, BERT, Custom NLP utilities
* **OCR:** Tesseract / Custom extractor (`extract_ocr.py`)

---

## 📂 Project Structure

```
FinanceBuddy/
│── myapp/
│   ├── finance/               # Core finance app
│   │   ├── templates/         # HTML templates
│   │   ├── migrations/        # DB migrations
│   │   ├── management/        # Custom commands
│   │   ├── ml_utils.py        # ML helper functions
│   │   ├── nlp_utils.py       # NLP utilities
│   │   ├── train_bert_intent_classifier.py
│   │   ├── views.py           # Django views
│   │   ├── models.py          # Django models
│   ├── models/                # Pre-trained ML models
│   ├── static/                # Static assets
│   ├── templates/             # Global templates
│   ├── manage.py              # Django entry point
│── Output/                    # Screenshots / Reports
│── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/FinanceBuddy.git
cd FinanceBuddy/myapp
```

2. **Create virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

3. **Run migrations**

```bash
python manage.py migrate
```

4. **Create superuser**

```bash
python manage.py createsuperuser
```

5. **Start server**

```bash
python manage.py runserver
```

Visit 👉 `http://127.0.0.1:8000/`

---

## ▶️ Usage

* **Dashboard:** Get an overview of expenses, income & investments
* **Bills:** Upload or scan bills (OCR auto extraction supported)
* **Chatbot:** Ask queries like *"Show me this month’s expenses"*
* **Reports:** Visualize spending, compare categories, download reports

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

MIT License – Feel free to use & modify.

---



