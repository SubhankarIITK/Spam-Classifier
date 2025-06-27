# 🛡️ Email/SMS Spam Classifier (with Streamlit)

A modern, interactive **Streamlit web app** that classifies text messages as **Spam** or **Not Spam** using a machine learning model trained on SMS data with **TF-IDF vectorization**.

---

## 🚀 Live Demo

> Coming soon — deploy this app on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🧠 Features

- 🧹 Preprocessing: lowercase, tokenization, stopword removal, stemming
- 🔠 Vectorization: `TfidfVectorizer` from `scikit-learn`
- 🤖 ML Model: Trained with Logistic Regression / Naive Bayes
- 🎨 Clean, responsive Streamlit UI with dark/light theme toggle
- 📊 Displays confidence score for each prediction
- ✅ `.gitignore` to exclude virtual envs and temp files

---

## 📂 Folder Structure

spam-classifier/
├── app.py ← Streamlit web app
├── model.pkl ← Trained ML model
├── vectorizer.pkl ← TF-IDF vectorizer
├── requirements.txt ← Project dependencies
├── .gitignore ← Git ignore rules
└── README.md ← This file

