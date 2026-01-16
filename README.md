# Telegram Reviews Sentiment Analysis

A Streamlit-based web application for analyzing sentiment in Telegram app reviews using machine learning.

## Features

- **Single Prediction**: Analyze individual review texts in real-time
- **Batch Prediction**: Process multiple reviews from CSV files
- **Indonesian Language Support**: Optimized for Indonesian text with slang normalization and stemming
- **Visual Analytics**: Interactive charts showing sentiment distribution and probability scores
- **Export Results**: Download batch prediction results as CSV

## Model Details

- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF
- **Classes**: Negative, Neutral, Positive
- **Preprocessing**: Slang normalization, stemming (Sastrawi), negation handling

## Usage

### Single Prediction
1. Select "Single Prediction" mode
2. Enter review text
3. Click "Analyze"

### Batch Prediction
1. Select "Batch Prediction" mode
2. Upload CSV file with `text` or `review` column
3. Click "Start Prediction"
4. Download results

## Requirements

- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- plotly
- Sastrawi
- joblib
