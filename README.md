

# 💸 Personal Finance Assistance System (FinanceBuddy)

An AI-powered **personal finance assistant** built with Django that simplifies financial tracking.
It integrates **OCR, NLP, Speech Recognition, and Machine Learning** to help users **log expenses, manage budgets, detect anomalies, forecast ROI, and interact via a chatbot**.

---

## 🚀 Features

* 💰 **Expense & Income Tracking** – Log daily transactions via text, receipts, or voice
* 🧾 **OCR Receipt Scanning** – Extract details from bills using **EasyOCR, PyTesseract, LayoutLM**
* 🎙 **Voice Input** – Add expenses hands-free with speech-to-text (SpeechRecognition)
* 🤖 **AI Chatbot** – Conversational finance queries with **DistilBERT**
* 📊 **Smart Analysis** – Interactive dashboards (Plotly) with category breakdowns
* 🧠 **Machine Learning Models**

  * ROI Prediction – **Linear Regression**
  * Anomaly Detection – **Isolation Forest**
  * Automated Categorization – **Naïve Bayes**
* 🔔 **Reminders & Insights** – Alerts for unusual expenses and predictive recommendations

---

## 🛠️ Tech Stack

* **Backend:** Django (Python 3.9, Django 4.2)
* **Frontend:** HTML, CSS (Tailwind), Django Templates
* **Database:** MySQL 8.0
* **NLP:** DistilBERT (Hugging Face)
* **OCR:** EasyOCR, PyTesseract, LayoutLM
* **ML:** Scikit-learn (Naïve Bayes, Isolation Forest, Linear Regression)
* **Visualization:** Plotly
* **Speech Recognition:** SpeechRecognition library
* **IDE:** VS Code
* **OS:** Windows 11

---

## 📂 Project Structure

```
FinanceBuddy/
│── finance/                   # Core app
│   ├── templates/             # HTML templates
│   ├── migrations/            # DB migrations
│   ├── management/            # Custom commands
│   ├── ml_utils.py            # ML helper functions
│   ├── nlp_utils.py           # NLP utilities
│   ├── views.py               # Django views
│   ├── models.py              # Django models
│── models/                    # Pre-trained ML models
│── static/                    # CSS, JS, images
│── Output/                    # Screenshots / Reports
│── requirements.txt           # Dependencies
│── manage.py                  # Django entry point
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/FinanceBuddy.git
   cd FinanceBuddy
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

* **Dashboard** – Get an overview of expenses, income, and investments
* **Bills** – Upload receipts for auto-extraction (OCR)
* **Chatbot** – Ask queries like *"Show me last week’s food expenses"*
* **Voice Input** – Speak expenses directly instead of typing
* **Reports** – Visualize spending, detect anomalies, and forecast ROI

---

## 🔮 Future Enhancements

* Better speech recognition with **Whisper / DeepSpeech**
* Improved OCR for low-quality receipts
* **Multilingual support** for broader accessibility
* **Bank API integration** for real-time transactions
* Mobile app via **Flutter/React Native**
* Advanced ML models (Autoencoders, XGBoost) for prediction
* Stronger security with **2FA & encryption**

---

## 🤝 Contributing

Contributions are welcome! Please fork this repo, make changes, and submit a PR.

---

## 📜 License

MIT License – free to use and modify.

---

## 👨‍💻 Author

* **Prasanth K**


---
