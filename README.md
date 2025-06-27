# 🛡️ Advanced Email/SMS Spam Classifier

An intelligent, feature-rich spam detection application built with Streamlit and Machine Learning. This application provides real-time spam classification with detailed text analysis, batch processing capabilities, and an intuitive user interface.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🔍 **Core Functionality**
- **Real-time Spam Detection**: Instant classification of email/SMS messages
- **High Accuracy**: Machine Learning model with TF-IDF vectorization
- **Confidence Scoring**: Get prediction confidence percentages
- **Text Preprocessing**: Advanced text cleaning with stemming and stopword removal

### 📊 **Advanced Analysis**
- **Detailed Text Metrics**: Word count, character count, uppercase ratio analysis
- **Spam Keyword Detection**: Identifies common spam indicators
- **URL & Contact Detection**: Finds URLs, phone numbers, and email addresses
- **Real-time Statistics**: Live metrics as you type

### 🚀 **Batch Processing**
- **Multiple Message Processing**: Analyze dozens of messages simultaneously
- **Progress Tracking**: Real-time progress bars and status updates
- **Visual Results**: Interactive pie charts and summary statistics
- **Export Ready**: Results displayed in clean, exportable tables

### 🎨 **User Experience**
- **Modern UI**: Professional design with gradients and animations
- **Responsive Layout**: Optimized for different screen sizes
- **Interactive Dashboard**: Sidebar with model info and statistics
- **Prediction History**: Track all your classifications with timestamps
- **Quick Examples**: Pre-loaded test messages for instant trials

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (first run will handle this automatically)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Prepare your model files**
   - Place your trained `model.pkl` file in the root directory
   - Place your `vectorizer.pkl` file in the root directory

5. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
spam-classifier/
│
├── app.py                 # Main application file
├── model.pkl             # Trained ML model (you need to provide this)
├── vectorizer.pkl        # TF-IDF vectorizer (you need to provide this)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
│
├── data/                # Optional: Training data directory
│   ├── spam.csv
│   └── ham.csv
│
├── models/              # Optional: Model training scripts
│   ├── train_model.py
│   └── evaluate_model.py
│
└── screenshots/         # Optional: App screenshots
    ├── main_interface.png
    ├── batch_processing.png
    └── analysis_view.png
```

## 📦 Dependencies

```txt
streamlit >= 1.28.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
nltk >= 3.6
plotly >= 5.0.0
pickle-mixin >= 1.0.2
```

## 🎯 Usage

### Single Message Classification

1. **Enter your message** in the text area
2. **Click "🔍 Analyze Message"** button
3. **View results** with confidence score and detailed analysis

### Batch Processing

1. **Enable batch mode** by checking the "📚 Batch Processing" checkbox
2. **Enter multiple messages** (one per line) in the batch text area
3. **Click "🚀 Process Batch"** to analyze all messages
4. **Review results** in the table and visualization

### Example Messages

**Spam Example:**
```
CONGRATULATIONS! You've won $1000! Click here NOW to claim your prize! Limited time offer!
```

**Legitimate Example:**
```
Hey, are we still meeting for lunch tomorrow at 12 PM? Let me know if you need to reschedule.
```

## 🔧 Configuration

### Model Requirements

Your model files should be:
- **model.pkl**: A trained scikit-learn classifier (supports `.predict()` and `.predict_proba()`)
- **vectorizer.pkl**: A fitted TF-IDF vectorizer from scikit-learn

### Customization Options

1. **Spam Keywords**: Edit the `get_spam_keywords()` function in `app.py`
2. **Text Analysis**: Modify the `analyze_text()` function for custom metrics
3. **Styling**: Update the CSS in the `st.markdown()` sections
4. **Model Parameters**: Adjust preprocessing in `transform_text()` function

## 📊 Model Training (Optional)

If you need to train your own model:

```python
# Example training script
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

# Load your data
data = pd.read_csv('spam_data.csv')

# Create and train model
vectorizer = TfidfVectorizer(max_features=3000)
model = MultinomialNB()

X_train_tfidf = vectorizer.fit_transform(data['text'])
model.fit(X_train_tfidf, data['label'])

# Save models
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --logger.level=debug
```

## 📈 Performance

- **Response Time**: < 100ms for single message classification
- **Batch Processing**: ~50-100 messages per second
- **Memory Usage**: ~50MB for model loading
- **Accuracy**: Depends on your trained model (typically 95%+ for good datasets)

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages with `pip install package_name`
2. **Model files not found**: Ensure `model.pkl` and `vectorizer.pkl` are in the root directory
3. **NLTK data missing**: Run the NLTK downloads in the prerequisites section
4. **Batch processing not working**: Make sure to check the batch processing checkbox
5. **White text visibility**: Clear browser cache and refresh the page

### Error Messages

| Error | Solution |
|-------|----------|
| `FileNotFoundError: model.pkl` | Place your trained model file in the project root |
| `Invalid property 'colors'` | Update plotly version: `pip install plotly --upgrade` |
| `NLTK punkt not found` | Run `nltk.download('punkt')` |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web app framework
- **scikit-learn** - For machine learning tools
- **NLTK** - For natural language processing
- **Plotly** - For interactive visualizations
- **Open Source Community** - For inspiration and support

## 📞 Support

If you encounter any issues or have questions:

1. **Check the [Issues](https://github.com/yourusername/spam-classifier/issues)** section
2. **Create a new issue** if your problem isn't already reported
3. **Provide detailed information** including error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Integration with BERT/transformers
- [ ] **Multi-language Support**: Support for non-English text
- [ ] **API Endpoint**: REST API for external integrations
- [ ] **Real-time Learning**: Model updates based on user feedback
- [ ] **Email Integration**: Direct integration with email providers
- [ ] **Mobile App**: React Native or Flutter mobile version
- [ ] **Advanced Analytics**: Detailed reporting and trends
- [ ] **Custom Model Training**: In-app model training interface

---

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ by [Subhankar Sutradhar](https://github.com/Subhankar)
