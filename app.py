import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
import re

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Text analysis functions
def analyze_text(text):
    """Analyze text for various spam indicators"""
    analysis = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'number_count': len(re.findall(r'\d+', text)),
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'email_count': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'phone_count': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    }
    return analysis

def get_spam_keywords():
    """Common spam keywords"""
    return ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'limited', 'offer', 
            'click', 'buy', 'sale', 'discount', 'guarantee', 'money', 'credit', 
            'loan', 'debt', 'cheap', 'amazing', 'incredible']

def check_spam_keywords(text):
    """Check for spam keywords in text"""
    spam_keywords = get_spam_keywords()
    text_lower = text.lower()
    found_keywords = [keyword for keyword in spam_keywords if keyword in text_lower]
    return found_keywords

# Page config
st.set_page_config(
    page_title="Advanced Spam Classifier", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .spam-result {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .spam-detected {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .not-spam {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    .analysis-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        color: #212529;
    }
    
    .analysis-card h4 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .analysis-card p {
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
    
    .keyword-tag {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        font-size: 16px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #212529;
    }
    
    .sidebar-content h4 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-content p {
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Advanced Spam Classifier</h1>
    <p>Intelligent email and SMS spam detection with detailed analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    
    # Model info
    st.markdown("""
    <div class="sidebar-content">
        <h4>🤖 Model Information</h4>
        <p><strong>Algorithm:</strong> Machine Learning Classifier</p>
        <p><strong>Features:</strong> TF-IDF Vectorization</p>
        <p><strong>Preprocessing:</strong> Stemming & Stopword Removal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        spam_count = sum(1 for pred in st.session_state.prediction_history if pred['result'] == 'Spam')
        spam_percentage = (spam_count / total_predictions) * 100
        
        st.markdown(f"""
        <div class="sidebar-content">
            <h4>📈 Session Statistics</h4>
            <p><strong>Total Predictions:</strong> {total_predictions}</p>
            <p><strong>Spam Detected:</strong> {spam_count}</p>
            <p><strong>Spam Rate:</strong> {spam_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick examples
    st.markdown("## 💡 Quick Examples")
    example_texts = {
        "🎁 Spam Example": "CONGRATULATIONS! You've won $1000! Click here NOW to claim your prize! Limited time offer!",
        "✅ Normal Example": "Hey, are we still meeting for lunch tomorrow at 12 PM? Let me know if you need to reschedule.",
        "📱 SMS Example": "Your package has been delivered to your front door. Thank you for choosing our service!"
    }
    
    for label, text in example_texts.items():
        if st.button(label, key=f"example_{label}"):
            st.session_state.example_text = text

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Message for Analysis")
    
    # Use example text if selected
    default_text = st.session_state.get('example_text', '')
    
    input_sms = st.text_area(
        "Message Content",
        value=default_text,
        height=150,
        placeholder="Type or paste your email/SMS text here for spam detection...",
        help="Enter the complete message content for accurate classification"
    )
    
    # Clear example text after use
    if 'example_text' in st.session_state:
        del st.session_state.example_text
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        show_analysis = st.checkbox("📊 Detailed Analysis", value=True)
    with col_b:
        show_keywords = st.checkbox("🔍 Keyword Detection", value=True)
    with col_c:
        batch_mode = st.checkbox("📚 Batch Processing", value=False)

with col2:
    if input_sms:
        # Real-time text analysis
        analysis = analyze_text(input_sms)
        
        st.markdown("### 📋 Text Metrics")
        
        metrics_html = f"""
        <div class="metric-card">
            <span style="color: #495057;"><strong>Words:</strong> {analysis['word_count']}</span><br>
            <span style="color: #495057;"><strong>Characters:</strong> {analysis['char_count']}</span><br>
            <span style="color: #495057;"><strong>Uppercase %:</strong> {analysis['uppercase_ratio']:.1%}</span><br>
            <span style="color: #495057;"><strong>Exclamations:</strong> {analysis['exclamation_count']}</span><br>
            <span style="color: #495057;"><strong>URLs:</strong> {analysis['url_count']}</span><br>
            <span style="color: #495057;"><strong>Phone Numbers:</strong> {analysis['phone_count']}</span>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

# Batch processing mode
if batch_mode:
    st.markdown("### 📚 Batch Processing")
    batch_input = st.text_area(
        "Enter multiple messages (one per line)",
        height=200,
        placeholder="Message 1\nMessage 2\nMessage 3...",
        key="batch_input"
    )
    
    if st.button('🚀 Process Batch', key="batch_process_btn"):
        if batch_input and batch_input.strip():
            messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
            
            if messages:
                batch_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, message in enumerate(messages):
                    status_text.text(f'Processing message {i+1} of {len(messages)}...')
                    
                    try:
                        transformed_sms = transform_text(message)
                        vector_input = tfidf.transform([transformed_sms])
                        result = model.predict(vector_input)[0]
                        probability = model.predict_proba(vector_input)[0][result]
                        
                        batch_results.append({
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Classification': 'Spam' if result == 1 else 'Not Spam',
                            'Confidence': f"{probability:.2%}"
                        })
                    except Exception as e:
                        st.error(f"Error processing message {i+1}: {str(e)}")
                        continue
                    
                    progress_bar.progress((i + 1) / len(messages))
                
                status_text.text('Processing complete!')
                
                if batch_results:
                    st.markdown("### 📊 Batch Results")
                    df = pd.DataFrame(batch_results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary chart
                    spam_count = sum(1 for r in batch_results if r['Classification'] == 'Spam')
                    not_spam_count = len(batch_results) - spam_count
                    
                    if spam_count > 0 or not_spam_count > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Not Spam', 'Spam'],
                            values=[not_spam_count, spam_count],
                            marker_colors=['#51cf66', '#ff6b6b']
                        )])
                        fig.update_layout(title="Batch Classification Results")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary stats
                        st.markdown(f"""
                        **Summary Statistics:**
                        - Total Messages: {len(batch_results)}
                        - Spam Detected: {spam_count}
                        - Not Spam: {not_spam_count}
                        - Spam Rate: {(spam_count/len(batch_results)*100):.1f}%
                        """)
                else:
                    st.error("No messages could be processed successfully.")
            else:
                st.warning("No valid messages found. Please enter at least one message.")
        else:
            st.warning("Please enter messages to process.")

# Single message prediction
if st.button('🔍 Analyze Message', type="primary", use_container_width=True):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        with st.spinner('🔄 Analyzing message...'):
            time.sleep(0.5)  # Add slight delay for better UX
            
            # Preprocess
            transformed_sms = transform_text(input_sms)
            
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # Predict
            result = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0][result]
            
            # Store in history
            prediction_record = {
                'timestamp': datetime.now(),
                'message': input_sms[:100] + '...' if len(input_sms) > 100 else input_sms,
                'result': 'Spam' if result == 1 else 'Not Spam',
                'confidence': probability
            }
            st.session_state.prediction_history.append(prediction_record)
            
            # Display result
            if result == 1:
                st.markdown(f"""
                <div class="spam-result spam-detected">
                    <h2>🚫 SPAM DETECTED</h2>
                    <p style="font-size: 1.2rem;">Confidence: {probability:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="spam-result not-spam">
                    <h2>✅ LEGITIMATE MESSAGE</h2>
                    <p style="font-size: 1.2rem;">Confidence: {probability:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis
            if show_analysis:
                st.markdown("### 📊 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis = analyze_text(input_sms)
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>📈 Text Statistics</h4>
                        <p><strong>Word Count:</strong> {analysis['word_count']}</p>
                        <p><strong>Character Count:</strong> {analysis['char_count']}</p>
                        <p><strong>Uppercase Ratio:</strong> {analysis['uppercase_ratio']:.1%}</p>
                        <p><strong>Special Characters:</strong> {analysis['exclamation_count']} exclamations, {analysis['question_count']} questions</p>
                        <p><strong>Numbers Found:</strong> {analysis['number_count']}</p>
                        <p><strong>URLs/Links:</strong> {analysis['url_count']}</p>
                        <p><strong>Phone Numbers:</strong> {analysis['phone_count']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if show_keywords:
                        spam_keywords_found = check_spam_keywords(input_sms)
                        if spam_keywords_found:
                            keywords_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in spam_keywords_found])
                            st.markdown(f"""
                            <div class="analysis-card">
                                <h4>🎯 Spam Keywords Detected</h4>
                                {keywords_html}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="analysis-card">
                                <h4>🎯 Spam Keywords</h4>
                                <p style="color: green;">✅ No common spam keywords detected</p>
                            </div>
                            """, unsafe_allow_html=True)

# Prediction history
if st.session_state.prediction_history:
    st.markdown("### 📚 Recent Predictions")
    
    # Create DataFrame for history
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['timestamp'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
    
    # Display recent predictions (last 10)
    recent_history = history_df.tail(10).iloc[::-1]  # Reverse to show newest first
    
    for _, row in recent_history.iterrows():
        status_color = "#ff6b6b" if row['result'] == 'Spam' else "#51cf66"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <strong style="color: #212529;">{row['result']}</strong> <span style="color: #6c757d;">({row['confidence']:.1%}) - {row['timestamp']}</span>
                    <br><em style="color: #495057;">{row['message']}</em>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button('🗑️ Clear History'):
        st.session_state.prediction_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 <strong>Tip:</strong> For better accuracy, provide complete message content including subject lines for emails.</p>
    <p>🔒 Your messages are processed locally and not stored permanently.</p>
</div>
""", unsafe_allow_html=True)