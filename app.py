import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from typing import List, Dict

from data import get_spam_data, get_not_spam_data, SPAM_EMAILS, NOT_SPAM_EMAILS
from naive_bayes import NaiveBayesClassifier
from agents import (
    DataPreprocessorAgent,
    FeatureExtractorAgent,
    TrainerAgent,
    PredictorAgent,
    EvaluatorAgent
)

st.set_page_config(
    page_title="🚀 Spam Detection AI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .spam-indicator {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .not-spam-indicator {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 107, 107, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    
    .agent-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .success-msg {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-msg {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'training_stats' not in st.session_state:
        st.session_state.training_stats = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'agents_initialized' not in st.session_state:
        st.session_state.agents_initialized = False
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

init_session_state()


@st.cache_resource
def initialize_agents():
    data_agent = DataPreprocessorAgent()
    feature_agent = FeatureExtractorAgent()
    trainer_agent = TrainerAgent()
    predictor_agent = PredictorAgent()
    evaluator_agent = EvaluatorAgent()
    
    return {
        'data': data_agent,
        'feature': feature_agent,
        'trainer': trainer_agent,
        'predictor': predictor_agent,
        'evaluator': evaluator_agent
    }


def create_confusion_matrix_chart(cm: Dict) -> go.Figure:
    z = [[cm['true_positive'], cm['false_negative']],
         [cm['false_positive'], cm['true_negative']]]
    
    x = ['Predicted Spam', 'Predicted Not Spam']
    y = ['Actual Spam', 'Actual Not Spam']
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='RdYlGn',
        text=[[f'TP: {z[0][0]}', f'FN: {z[0][1]}'],
              [f'FP: {z[1][0]}', f'TN: {z[1][1]}']],
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='%{y} vs %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def create_metrics_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccb'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_word_cloud_chart(top_words: List) -> go.Figure:
    if not top_words:
        return None
    
    words = [w[0] for w in top_words[:10]]
    ratios = [w[1] for w in top_words[:10]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=ratios,
            y=words,
            orientation='h',
            marker=dict(
                color=ratios,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='Spam Ratio')
            )
        )
    ])
    
    fig.update_layout(
        title='Top Spam Indicator Words',
        xaxis_title='Spam Likelihood Ratio',
        yaxis_title='Word',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_data_distribution_chart() -> go.Figure:
    labels = ['Spam', 'Not Spam']
    values = [len(SPAM_EMAILS), len(NOT_SPAM_EMAILS)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=['#ff6b6b', '#51cf66']
    )])
    
    fig.update_layout(
        title='Dataset Distribution',
        height=300
    )
    
    return fig


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>📧 Spam Detector</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        page = st.radio(
            "🧭 Navigation",
            ["🏠 Home", "📊 Data Explorer", "🎓 Train Model", 
             "🔮 Predict", "📈 Evaluation", "🤖 AI Agents", "ℹ️ About"],
            index=0
        )
        
        st.divider()
        
        st.markdown("### 📊 Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
            if st.session_state.training_stats:
                st.metric("Vocabulary Size", 
                         st.session_state.training_stats.get('vocabulary_size', 0))
        else:
            st.warning("⚠️ Model Not Trained")
            st.caption("Go to 'Train Model' to train")
        
        st.divider()
        
        st.markdown("### 📧 Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam", len(SPAM_EMAILS))
        with col2:
            st.metric("Not Spam", len(NOT_SPAM_EMAILS))
        
        return page


def render_home():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Spam Detection with AI Agents</h1>
        <p>Intelligent Email Classification using Naive Bayes & CAMEL AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📊</div>
            <div class="metric-label">Explore Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🎓</div>
            <div class="metric-label">Train Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔮</div>
            <div class="metric-label">Predict Spam</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📈</div>
            <div class="metric-label">Evaluate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🧠 Naive Bayes Algorithm
        - Custom implementation from scratch
        - Laplace smoothing for unknown words
        - Probability-based classification
        
        #### 🤖 Multi-Agent Architecture
        - **Data Agent**: Preprocessing & validation
        - **Feature Agent**: Pattern extraction
        - **Trainer Agent**: Model training
        - **Predictor Agent**: Classification
        - **Evaluator Agent**: Performance analysis
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Interactive Dashboard
        - Real-time predictions
        - Visual analytics
        - Confusion matrix visualization
        
        #### 🎯 High Accuracy
        - Trained on 130 emails
        - Balanced dataset
        - 90%+ accuracy achievable
        """)
    
    st.markdown("---")
    st.markdown("### 🎮 Quick Demo")
    
    demo_email = st.text_input(
        "Try it out! Enter an email to check:",
        placeholder="Example: Congratulations! You won $1,000,000!"
    )
    
    if demo_email and st.session_state.model_trained:
        result = st.session_state.classifier.predict_single(demo_email)
        pred, probs = result
        
        if pred == "spam":
            st.markdown("""
            <div class="spam-indicator">
                🚨 SPAM DETECTED!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="not-spam-indicator">
                ✅ NOT SPAM
            </div>
            """, unsafe_allow_html=True)
        
        st.metric("Confidence", f"{max(probs.values())*100:.1f}%")
    elif demo_email:
        st.warning("⚠️ Please train the model first! Go to 'Train Model' section.")


def render_data_explorer():
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the email dataset used for training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📈 Distribution")
        fig = create_data_distribution_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Statistics")
        st.metric("Total Emails", len(SPAM_EMAILS) + len(NOT_SPAM_EMAILS))
        st.metric("Spam Emails", len(SPAM_EMAILS))
        st.metric("Not Spam Emails", len(NOT_SPAM_EMAILS))
    
    with col2:
        st.markdown("### 📧 Sample Emails")
        
        tab1, tab2 = st.tabs(["🚨 Spam Emails", "✅ Not Spam Emails"])
        
        with tab1:
            spam_df = pd.DataFrame(SPAM_EMAILS)
            spam_df.index = range(1, len(spam_df) + 1)
            st.dataframe(
                spam_df,
                use_container_width=True,
                height=400
            )
        
        with tab2:
            not_spam_df = pd.DataFrame(NOT_SPAM_EMAILS)
            not_spam_df.index = range(1, len(not_spam_df) + 1)
            st.dataframe(
                not_spam_df,
                use_container_width=True,
                height=400
            )
    
    st.markdown("---")
    st.markdown("### 📏 Email Length Analysis")
    
    spam_lengths = [len(e['email']) for e in SPAM_EMAILS]
    not_spam_lengths = [len(e['email']) for e in NOT_SPAM_EMAILS]
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=spam_lengths, name='Spam', marker_color='#ff6b6b'))
    fig.add_trace(go.Box(y=not_spam_lengths, name='Not Spam', marker_color='#51cf66'))
    fig.update_layout(
        title='Email Length Distribution',
        yaxis_title='Character Count',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_train_model():
    st.markdown("## 🎓 Train Naive Bayes Model")
    
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        smoothing = st.slider(
            "Laplace Smoothing",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Smoothing parameter for handling unknown words"
        )
    
    with col2:
        test_size = st.slider(
            "Test Split %",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data for testing"
        )
    
    with col3:
        min_word_len = st.slider(
            "Min Word Length",
            min_value=1,
            max_value=5,
            value=2,
            help="Minimum word length to consider"
        )
    
    st.markdown("---")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📦 Initializing agents...")
            progress.progress(10)
            agents = initialize_agents()
            time.sleep(0.3)
            
            status.text("📊 Loading data...")
            progress.progress(20)
            spam_data = get_spam_data()
            not_spam_data = get_not_spam_data()
            time.sleep(0.3)
            
            status.text("✅ Validating data...")
            progress.progress(30)
            validation = agents['data'].load_and_validate_data(spam_data, not_spam_data)
            time.sleep(0.3)
            
            status.text("📂 Splitting data...")
            progress.progress(40)
            train_emails, train_labels, test_emails, test_labels = agents['data'].split_data(
                validation["valid_emails"],
                validation["labels"],
                test_size=test_size/100
            )
            time.sleep(0.3)
            
            status.text("🎓 Training classifier...")
            progress.progress(60)
            
            classifier = NaiveBayesClassifier(smoothing=smoothing)
            classifier.min_word_length = min_word_len
            training_stats = classifier.fit(train_emails, train_labels)
            time.sleep(0.3)
            
            status.text("📈 Evaluating model...")
            progress.progress(80)
            
            predictions = []
            for email in test_emails:
                pred, probs = classifier.predict_single(email)
                predictions.append({
                    "prediction": pred,
                    "probabilities": probs,
                    "confidence": max(probs.values()) * 100
                })
            
            tp = tn = fp = fn = 0
            for pred, true in zip(predictions, test_labels):
                if pred["prediction"] == "spam" and true == "spam":
                    tp += 1
                elif pred["prediction"] == "not_spam" and true == "not_spam":
                    tn += 1
                elif pred["prediction"] == "spam" and true == "not_spam":
                    fp += 1
                else:
                    fn += 1
            
            total = tp + tn + fp + fn
            evaluation_results = {
                "confusion_matrix": {
                    "true_positive": tp,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn
                },
                "metrics": {
                    "accuracy": (tp + tn) / total if total > 0 else 0,
                    "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                    "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                    "f1_score": 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
                }
            }
            
            st.session_state.classifier = classifier
            st.session_state.model_trained = True
            st.session_state.training_stats = training_stats
            st.session_state.training_stats['top_spam_indicators'] = classifier.get_top_spam_words(10)
            st.session_state.evaluation_results = evaluation_results
            
            progress.progress(100)
            status.text("✅ Training complete!")
            time.sleep(0.5)
            
            st.success("🎉 Model trained successfully!")

    if st.session_state.model_trained and st.session_state.training_stats:
        st.markdown("---")
        st.markdown("### 📊 Training Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Vocabulary Size",
                st.session_state.training_stats.get('vocabulary_size', 0)
            )
        
        with col2:
            st.metric(
                "Training Samples",
                st.session_state.training_stats.get('total_documents', 0)
            )
        
        with col3:
            priors = st.session_state.training_stats.get('class_priors', {})
            st.metric(
                "Spam Prior",
                f"{priors.get('spam', 0)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Not Spam Prior",
                f"{priors.get('not_spam', 0)*100:.1f}%"
            )
        
        st.markdown("### 🔥 Top Spam Indicator Words")
        top_words = st.session_state.training_stats.get('top_spam_indicators', [])
        if top_words:
            fig = create_word_cloud_chart(top_words)
            st.plotly_chart(fig, use_container_width=True)


def render_predict():
    st.markdown("## 🔮 Spam Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained! Please train the model first.")
        if st.button("Go to Training"):
            st.session_state.page = "🎓 Train Model"
        return
    
    st.markdown("### 📧 Check Single Email")
    
    email_input = st.text_area(
        "Enter email text to analyze:",
        height=150,
        placeholder="Type or paste an email here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if predict_btn and email_input:
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            pred, probs = st.session_state.classifier.predict_single(email_input)
            confidence = max(probs.values()) * 100
            
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if pred == "spam":
                    st.markdown("""
                    <div class="spam-indicator">
                        🚨 SPAM DETECTED!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="not-spam-indicator">
                        ✅ NOT SPAM - Safe Email
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                fig = create_metrics_gauge(confidence/100, "Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Spam', 'Not Spam'],
                'Probability': [probs.get('spam', 0)*100, probs.get('not_spam', 0)*100]
            })
            
            fig = px.bar(
                prob_df, 
                x='Class', 
                y='Probability',
                color='Class',
                color_discrete_map={'Spam': '#ff6b6b', 'Not Spam': '#51cf66'}
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔤 Influential Words")
            words = st.session_state.classifier.preprocess_text(email_input)
            word_analysis = []
            
            for word in set(words):
                if word in st.session_state.classifier.vocabulary:
                    spam_prob = st.session_state.classifier.word_likelihoods.get('spam', {}).get(word, 0)
                    not_spam_prob = st.session_state.classifier.word_likelihoods.get('not_spam', {}).get(word, 0)
                    if not_spam_prob > 0:
                        ratio = spam_prob / not_spam_prob
                        word_analysis.append({
                            'Word': word,
                            'Spam Ratio': ratio,
                            'Indicates': '🚨 Spam' if ratio > 1 else '✅ Not Spam'
                        })
            
            if word_analysis:
                word_df = pd.DataFrame(word_analysis)
                word_df = word_df.sort_values('Spam Ratio', ascending=False).head(10)
                st.dataframe(word_df, use_container_width=True)
            
            st.session_state.prediction_history.append({
                'email': email_input[:50] + '...' if len(email_input) > 50 else email_input,
                'prediction': pred,
                'confidence': confidence
            })
    
    st.markdown("---")
    st.markdown("### 📦 Batch Prediction")
    
    with st.expander("Upload multiple emails"):
        uploaded_file = st.file_uploader("Upload a text file (one email per line)", type=['txt'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            emails = [e.strip() for e in content.split('\n') if e.strip()]
            
            st.info(f"Found {len(emails)} emails")
            
            if st.button("Analyze All"):
                results = []
                progress = st.progress(0)
                
                for i, email in enumerate(emails):
                    pred, probs = st.session_state.classifier.predict_single(email)
                    results.append({
                        'Email': email[:50] + '...',
                        'Prediction': pred,
                        'Confidence': f"{max(probs.values())*100:.1f}%"
                    })
                    progress.progress((i+1)/len(emails))
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                spam_count = sum(1 for r in results if r['Prediction'] == 'spam')
                st.metric("Spam Detected", f"{spam_count}/{len(results)}")
    
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.prediction_history[-10:])
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🧪 Try Sample Emails")
    
    sample_emails = {
        "🎰 Lottery Spam": "Congratulations! You've won $1,000,000 in our lottery! Click here to claim now!",
        "💊 Pharmacy Spam": "Buy cheap medications online! Viagra, Cialis at lowest prices! Order now!",
        "📧 Normal Email": "Hi John, can we schedule a meeting for tomorrow at 3pm? Let me know. Thanks!"
    }
    
    cols = st.columns(3)
    for i, (label, email) in enumerate(sample_emails.items()):
        with cols[i % 3]:
            if st.button(label, key=f"sample_{i}"):
                pred, probs = st.session_state.classifier.predict_single(email)
                st.info(f"**Email:** {email}")
                if pred == "spam":
                    st.error(f"🚨 SPAM ({max(probs.values())*100:.1f}%)")
                else:
                    st.success(f"✅ NOT SPAM ({max(probs.values())*100:.1f}%)")


def render_evaluation():
    st.markdown("## 📈 Model Evaluation")
    
    if not st.session_state.model_trained or not st.session_state.evaluation_results:
        st.warning("⚠️ No evaluation results. Please train the model first.")
        return
    
    results = st.session_state.evaluation_results
    metrics = results['metrics']
    cm = results['confusion_matrix']
    
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_metrics_gauge(metrics['accuracy'], 'Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_metrics_gauge(metrics['precision'], 'Precision')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_metrics_gauge(metrics['recall'], 'Recall')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = create_metrics_gauge(metrics['f1_score'], 'F1 Score')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_confusion_matrix_chart(cm)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Matrix Values")
        st.metric("True Positives (TP)", cm['true_positive'], 
                 help="Spam correctly identified as spam")
        st.metric("True Negatives (TN)", cm['true_negative'],
                 help="Not spam correctly identified as not spam")
        st.metric("False Positives (FP)", cm['false_positive'],
                 help="Not spam incorrectly marked as spam")
        st.metric("False Negatives (FN)", cm['false_negative'],
                 help="Spam incorrectly marked as not spam")
    
    st.markdown("---")
    st.markdown("### 📚 Understanding the Metrics")
    
    with st.expander("Click to learn about each metric"):
        st.markdown("""
        | Metric | Formula | Interpretation |
        |--------|---------|----------------|
        | **Accuracy** | (TP + TN) / Total | Overall correctness |
        | **Precision** | TP / (TP + FP) | How many predicted spam are actually spam |
        | **Recall** | TP / (TP + FN) | How many actual spam were detected |
        | **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
        
        ---
        
        **For Spam Detection:**
        - High **Precision** = Few false alarms (legitimate emails marked as spam)
        - High **Recall** = Few missed spam emails
        - **F1 Score** balances both concerns
        """)


def render_ai_agents():
    st.markdown("## 🤖 CAMEL AI Agents")
    st.markdown("Meet the intelligent agents powering this spam detection system!")
    
    agents_info = [
        {
            "name": "📦 Data Preprocessor Agent",
            "role": "Data Specialist",
            "tasks": [
                "Load and validate email data",
                "Check data quality",
                "Split data into train/test sets",
                "Generate data statistics"
            ],
            "color": "#667eea"
        },
        {
            "name": "🔍 Feature Extractor Agent", 
            "role": "Pattern Analyst",
            "tasks": [
                "Extract text features from emails",
                "Identify spam indicators",
                "Analyze word patterns",
                "Generate feature reports"
            ],
            "color": "#764ba2"
        },
        {
            "name": "🎓 Trainer Agent",
            "role": "ML Engineer",
            "tasks": [
                "Initialize Naive Bayes classifier",
                "Train model on data",
                "Calculate word probabilities",
                "Track training progress"
            ],
            "color": "#f093fb"
        },
        {
            "name": "🔮 Predictor Agent",
            "role": "Classification Expert",
            "tasks": [
                "Classify new emails",
                "Calculate confidence scores",
                "Identify influential words",
                "Explain predictions"
            ],
            "color": "#f5576c"
        },
        {
            "name": "📈 Evaluator Agent",
            "role": "Quality Analyst", 
            "tasks": [
                "Calculate accuracy metrics",
                "Generate confusion matrix",
                "Analyze model performance",
                "Suggest improvements"
            ],
            "color": "#4facfe"
        }
    ]
    
    for agent in agents_info:
        with st.expander(f"{agent['name']} - {agent['role']}", expanded=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"""
                <div style="
                    background: {agent['color']};
                    color: white;
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    font-size: 3rem;
                ">
                    {agent['name'].split()[0]}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Role:** {agent['role']}")
                st.markdown("**Responsibilities:**")
                for task in agent['tasks']:
                    st.markdown(f"- {task}")
    
    st.markdown("---")
    st.markdown("### 💬 Ask AI Agent")
    
    agent_choice = st.selectbox(
        "Select an agent to interact with:",
        ["Trainer Agent - Explain Naive Bayes",
         "Evaluator Agent - Analyze Performance",
         "Feature Agent - Explain Features"]
    )
    
    if st.button("🤖 Get AI Response"):
        with st.spinner("AI is thinking..."):
            agents = initialize_agents()
            
            if "Trainer" in agent_choice:
                response = agents['trainer'].explain_naive_bayes()
            elif "Evaluator" in agent_choice:
                if st.session_state.evaluation_results:
                    response = agents['evaluator'].get_ai_evaluation_analysis()
                else:
                    response = "Please train the model first to get evaluation analysis."
            else:
                response = "Feature extraction helps identify patterns like spam keywords, excessive punctuation, and urgency words."
            
            st.info(response)


def render_about():
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🚀 Spam Detection with CAMEL AI Agents
    
    This project demonstrates a complete spam detection system built with:
    
    #### 🧠 Technology Stack
    - **Naive Bayes Classifier**: Custom implementation from scratch
    - **CAMEL AI**: Multi-agent framework for intelligent processing
    - **Groq LLaMA 3.3 70B**: Large language model for AI insights
    - **Streamlit**: Interactive web dashboard
    - **Plotly**: Beautiful visualizations
    
    #### 📊 Dataset
    - **130 emails** (70 spam + 60 not spam)
    - Balanced dataset for fair training
    - Real-world spam patterns
    
    #### 🎯 How Naive Bayes Works
    
    1. **Training Phase:**
       - Calculate P(spam) and P(not_spam) - prior probabilities
       - Calculate P(word|spam) and P(word|not_spam) for each word
       
    2. **Prediction Phase:**
       - For a new email, calculate:
         - P(spam|words) ∝ P(spam) × Π P(word|spam)
         - P(not_spam|words) ∝ P(not_spam) × Π P(word|not_spam)
       - Choose the class with higher probability
    
    #### 🤖 Multi-Agent Architecture
    
    Each agent has a specific role:
    - **Data Agent**: Handles data preprocessing
    - **Feature Agent**: Extracts meaningful features
    - **Trainer Agent**: Trains the ML model
    - **Predictor Agent**: Makes predictions
    - **Evaluator Agent**: Evaluates performance
    
    ---
    
    ### 👨‍💻 Built with ❤️
    
    This project was created as a demonstration of combining traditional ML algorithms 
    with modern AI agent frameworks for intelligent spam detection.
    """)
    
    st.markdown("### 🏗️ System Architecture")
    
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    STREAMLIT UI                             │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
    │  │  Home   │ │  Data   │ │  Train  │ │ Predict │           │
    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
    └───────┼───────────┼───────────┼───────────┼─────────────────┘
            │           │           │           │
    ┌───────▼───────────▼───────────▼───────────▼─────────────────┐
    │                 CAMEL AI AGENTS LAYER                       │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
    │  │   Data   │ │ Feature  │ │ Trainer  │ │Predictor │       │
    │  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │       │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
    └───────┼────────────┼────────────┼────────────┼──────────────┘
            │            │            │            │
    ┌───────▼────────────▼────────────▼────────────▼──────────────┐
    │              NAIVE BAYES CLASSIFIER                         │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
    │  │ Vocabulary  │ │ Likelihoods │ │   Priors    │           │
    │  └─────────────┘ └─────────────┘ └─────────────┘           │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)


def main():
    page = render_sidebar()
    
    if page == "🏠 Home":
        render_home()
    elif page == "📊 Data Explorer":
        render_data_explorer()
    elif page == "🎓 Train Model":
        render_train_model()
    elif page == "🔮 Predict":
        render_predict()
    elif page == "📈 Evaluation":
        render_evaluation()
    elif page == "🤖 AI Agents":
        render_ai_agents()
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()