import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Sentiment Analysis - Telegram Reviews",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD STEMMER
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# DICTIONARY
SLANG_DICT = {'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'gx': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'tdk': 'tidak', 'bgt': 'banget', 'bgs': 'bagus', 'jgn': 'jangan', 'blm': 'belum', 'udh': 'sudah', 'udah': 'sudah', 'gimana': 'bagaimana', 'gmn': 'bagaimana', 'knp': 'kenapa', 'tp': 'tapi', 'tq': 'terima kasih', 'mksh': 'terima kasih', 'trmksh': 'terima kasih', 'krn': 'karena', 'krna': 'karena', 'emang': 'memang', 'emg': 'memang', 'bner': 'benar', 'bnr': 'benar', 'org': 'orang', 'jd': 'jadi', 'jdi': 'jadi', 'bs': 'bisa', 'bkn': 'bukan', 'sm': 'sama', 'dr': 'dari', 'utk': 'untuk', 'dgn': 'dengan', 'yg': 'yang', 'pd': 'pada', 'sdh': 'sudah', 'trs': 'terus', 'klo': 'kalau', 'kalo': 'kalau', 'spy': 'supaya', 'aja': 'saja', 'aj': 'saja', 'donk': 'dong', 'mantul': 'mantap', 'josss': 'bagus', 'top': 'bagus', 'jelek': 'buruk', 'parah': 'buruk', 'zonk': 'buruk', 'ampas': 'buruk'}
NEGATIONS = {'tidak', 'tak', 'nggak', 'nga', 'enggak', 'gak', 'gx', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang', 'minus', 'kecewa', 'mengecewakan', 'buruk', 'jelek', 'payah', 'ga', 'g', 'tdk', 'jgn', 'blm'}

# PREPROCESSING
def preprocess_text(text):
    """Complete text preprocessing"""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Normalize slang
    words = text.split()
    words = [SLANG_DICT.get(word, word) for word in words]
    
    # Stemming
    words = [stemmer.stem(word) for word in words]
    
    # Negation handling
    result = []
    negate = False
    count = 0
    window = 4
    
    for word in words:
        if word in NEGATIONS:
            result.append(word)
            negate = True
            count = 0
        elif negate and count < window:
            if word not in {'yang', 'ini', 'itu', 'dan', 'atau'}:
                result.append(f'NOT_{word}')
            else:
                result.append(word)
            count += 1
        else:
            result.append(word)
            negate = False
    
    return ' '.join(result)

# LOAD MODEL
@st.cache_resource
def load_model_artifacts():
    """Load model, thresholds, and class mapping"""
    try:
        model = joblib.load('sentiment_model.pkl')
        thresholds = joblib.load('thresholds.pkl')
        class_mapping_raw = joblib.load('class_mapping.pkl')
        
        # NORMALIZE CLASS MAPPING TO DICTIONARY
        if isinstance(class_mapping_raw, dict):
            class_mapping = {int(k): str(v) for k, v in class_mapping_raw.items()}
        elif isinstance(class_mapping_raw, (np.ndarray, list)):
            class_mapping = {i: str(cls) for i, cls in enumerate(class_mapping_raw)}
        else:
            st.warning("class_mapping format not recognized, using default")
            class_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        return model, thresholds, class_mapping
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("Make sure these files are in the same folder as app.py:")
        st.code("sentiment_model.pkl\nthresholds.pkl\nclass_mapping.pkl")
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# PREDICTION
def predict_sentiment(text, model, thresholds, class_mapping):
    """Predict sentiment with threshold tuning"""
    processed_text = preprocess_text(text)
    proba = model.predict_proba([processed_text])[0]
    adjusted_proba = proba / thresholds
    pred_class_idx = np.argmax(adjusted_proba)
    pred_class = class_mapping[pred_class_idx]
    confidence = adjusted_proba[pred_class_idx]
    
    probabilities = {class_mapping[i]: float(p) for i, p in enumerate(proba)}
    
    return {
        'prediction': pred_class,
        'confidence': float(confidence),
        'probabilities': probabilities,
        'processed_text': processed_text
    }

# STYLING
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #fff3cd; color: #856404; }
    </style>
""", unsafe_allow_html=True)

# MAIN APP
def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin: 0 0 3rem 0;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 3px solid #3498db;
    }
    </style>
    <p class="main-header">Telegram Reviews Sentiment Analysis</p>
    """, unsafe_allow_html=True)    
    # Load model
    model, thresholds, class_mapping = load_model_artifacts()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <h3><span class="material-icons" style="vertical-align: middle;">settings</span> Settings</h3>
        """, unsafe_allow_html=True)
        
        mode = st.radio("Select Mode:", ["Single Prediction", "Batch Prediction"])

        st.markdown("---")
        st.markdown("""
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <h3 style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
            <span class="material-icons">psychology</span>
            <span>Model Info</span>
        </h3>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="
            border-left: 4px solid #3498db;
            padding-left: 12px;
            margin-bottom: 16px;
        ">
            <div style="color: #6B7280; font-size: 14px; margin-bottom: 4px;">Classes</div>
            <div style="font-size: 16px; font-weight: 600;">Negative, Neutral, Positive</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="
            border-left: 4px solid #3498db;
            padding-left: 12px;
            margin-bottom: 16px;
        ">
            <div style="color: #6B7280; font-size: 14px; margin-bottom: 4px;">Algorithm</div>
            <div style="font-size: 16px; font-weight: 600;">Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="
            border-left: 4px solid #3498db;
            padding-left: 12px;
            margin-bottom: 16px;
        ">
            <div style="color: #6B7280; font-size: 14px; margin-bottom: 4px;">Features</div>
            <div style="font-size: 16px; font-weight: 600;">TF-IDF</div>
        </div>
        """, unsafe_allow_html=True)
    
    # SINGLE PREDICTION
    if mode == "Single Prediction":
        st.markdown("""
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <h3><span class="material-icons" style="vertical-align: middle;">edit_note</span> Enter Review Text</h3>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Type or paste review:",
            height=150,
            placeholder="Example: Aplikasinya bagus dan gampang dipakai!"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # predict_btn = st.button("Analyze", type="primary", use_container_width=True)
            st.markdown("""
            <style>
            button[kind="primary"] {
                background-color: #3498db !important;
                border-color: #3498db !important;
                color: white !important;
            }
            button[kind="primary"]:hover {
                background-color: #1a252f !important;
                border-color: #1a252f !important;
            }
            </style>
            """, unsafe_allow_html=True)

            predict_btn = st.button("Analyze", type="primary", use_container_width=True)
        
        if predict_btn and user_input.strip():
            with st.spinner("Analyzing..."):
                result = predict_sentiment(user_input, model, thresholds, class_mapping)
                
                st.markdown("---")
                st.markdown("""
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <h3><span class="material-icons" style="vertical-align: middle;">analytics</span> Analysis Results</h3>
                """, unsafe_allow_html=True)
                
                sentiment = result['prediction']
                confidence = result['confidence']
                
                sentiment_lower = sentiment.lower()
                if 'positif' in sentiment_lower or 'positive' in sentiment_lower:
                    box_class = 'positive'
                    sentiment_display = 'POSITIVE'
                    emoji = '<span class="material-icons">sentiment_satisfied_alt</span>'
                elif 'negatif' in sentiment_lower or 'negative' in sentiment_lower:
                    box_class = 'negative'
                    sentiment_display = 'NEGATIVE'
                    emoji = '<span class="material-icons">sentiment_dissatisfied</span>'
                else:
                    box_class = 'neutral'
                    sentiment_display = 'NEUTRAL'
                    emoji = '<span class="material-icons">sentiment_neutral</span>'
                
                st.markdown(f"""
                    <div class="sentiment-box {box_class}">
                        <h2>{emoji} {sentiment_display}</h2>
                        <p style="font-size: 1.2rem;">Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Probability Distribution")

                    prob_df = pd.DataFrame({
                        'Sentiment': list(result['probabilities'].keys()),
                        'Probability': list(result['probabilities'].values())
                    })

                    label_map = {'negatif': 'Negative', 'netral': 'Neutral', 'positif': 'Positive'}
                    color_map = {'Negative': '#EF4444', 'Neutral': '#FCD34D', 'Positive': '#10B981'}

                    prob_df['Sentiment'] = prob_df['Sentiment'].map(label_map)

                    colors = [color_map.get(sentiment, '#6B7280') for sentiment in prob_df['Sentiment']]

                    fig = px.bar(prob_df, x='Sentiment', y='Probability')
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(showlegend=False,  yaxis_title="Probability",  xaxis_title="",  height=400)

                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Probability Details")

                    label_map = {'negatif': 'Negative', 'netral': 'Neutral', 'positif': 'Positive'}
                    color_map = {'negatif': '#EF4444', 'netral': '#FCD34D', 'positif': '#10B981'}

                    for sent, prob in result['probabilities'].items():
                        color = color_map.get(sent, '#6B7280')
                        label = label_map.get(sent, sent.capitalize())
                        
                        st.markdown(f"""
                        <div style="
                            border-left: 4px solid {color};
                            padding-left: 12px;
                            margin-bottom: 16px;
                        ">
                            <div style="color: #6B7280; font-size: 14px; margin-bottom: 4px;">{label}</div>
                            <div style="font-size: 24px; font-weight: 600;">{prob:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)

        elif predict_btn:
            st.markdown("""
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px; display: flex; align-items: center; gap: 10px;">
                <span class="material-icons" style="color: #856404;">warning</span>
                <span style="color: #856404;">Please enter text first!</span>
            </div>
            """, unsafe_allow_html=True)
    
    # BATCH PREDICTION
    else:
        st.markdown("""
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <h3><span class="material-icons" style="vertical-align: middle;">upload_file</span> Upload CSV File</h3>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload CSV (must have 'text' or 'review' column)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find text column
                text_col = None
                for col in ['text', 'review', 'ulasan', 'Ulasan', 'Text', 'Review']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    st.markdown("""
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; border-radius: 4px; display: flex; align-items: center; gap: 10px;">
                        <span class="material-icons" style="color: #721c24;">error</span>
                        <span style="color: #721c24;">No 'text' or 'review' column found in CSV!</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                st.markdown(f"""
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; border-radius: 4px; display: flex; align-items: center; gap: 10px;">
                    <span class="material-icons" style="color: #155724;">check_circle</span>
                    <span style="color: #155724;">{len(df)} rows of data successfully uploaded</span>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Start Prediction", type="primary"):
                    with st.spinner("Processing..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(df[text_col]):
                            result = predict_sentiment(text, model, thresholds, class_mapping)
                            results.append({
                                'original_text': text,
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                **{f'prob_{k}': v for k, v in result['probabilities'].items()}
                            })
                            progress_bar.progress((idx + 1) / len(df))
                        
                        result_df = pd.DataFrame(results)
                        
                        st.markdown("""
                        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; border-radius: 4px; display: flex; align-items: center; gap: 10px;">
                            <span class="material-icons" style="color: #155724;">check_circle</span>
                            <span style="color: #155724;">Prediction completed!</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Summary
                        st.markdown("""
                        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        <h3><span class="material-icons" style="vertical-align: middle;">summarize</span> Summary</h3>
                        """, unsafe_allow_html=True)

                        sentiment_counts = result_df['prediction'].value_counts()

                        label_map = {'negatif': 'Negative', 'netral': 'Neutral', 'positif': 'Positive'}
                        color_map = {'negatif': '#EF4444', 'netral': '#FCD34D', 'positif': '#10B981'}

                        cols = st.columns(len(sentiment_counts))
                        for idx, (sent, count) in enumerate(sentiment_counts.items()):
                            with cols[idx]:
                                pct = count / len(result_df) * 100
                                color = color_map.get(sent, '#6B7280')
                                label = label_map.get(sent, sent.capitalize())
                                
                                st.markdown(f"""
                                <div style="
                                    border-left: 4px solid {color};
                                    padding-left: 12px;
                                ">
                                    <div style="color: #6B7280; font-size: 14px; margin-bottom: 4px;">{label}</div>
                                    <div style="font-size: 32px; font-weight: 600; margin-bottom: 4px;">{count}</div>
                                    <div style="color: #6B7280; font-size: 14px;">{pct:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)

                        labels = [label_map.get(sent, sent.capitalize()) for sent in sentiment_counts.index]
                        colors = [color_map.get(sent, '#6B7280') for sent in sentiment_counts.index]

                        fig = px.pie(values=sentiment_counts.values, names=labels, title="Sentiment Distribution")
                        fig.update_traces(marker=dict(colors=colors), textfont_size=18)
                        fig.update_layout(legend=dict(font=dict(size=16)))

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("""
                        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        <h3><span class="material-icons" style="vertical-align: middle;">list_alt</span> Detailed Results</h3>
                        """, unsafe_allow_html=True)
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇ Download Results (CSV)",
                            data=csv,
                            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.markdown(f"""
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; border-radius: 4px; display: flex; align-items: center; gap: 10px;">
                    <span class="material-icons" style="color: #721c24;">error</span>
                    <span style="color: #721c24;">Error: {str(e)}</span>
                </div>
                """, unsafe_allow_html=True)
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
