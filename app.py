import streamlit as st
import pandas as pd
import json
import logging
from sentiment_model import SentimentAnalyzer
import streamlit.components.v1 as components

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# App title and layout
st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis Dashboard")

# Single text analysis
st.header("Analyze Single Text")
with st.form("text_analysis"):
    text = st.text_area("Enter text for sentiment analysis", height=150, placeholder="Type your text here...")
    submit_text = st.form_submit_button("Analyze")
    if submit_text:
        try:
            if not text.strip():
                st.error("Please enter some text")
            else:
                result = analyzer.analyze_text(text)
                st.subheader("Analysis Results")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.json(result)
                with col2:
                    st.subheader("Sentiment Scores")
                    components.html(
                        f"""
                        <canvas id="chart"></canvas>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                        <script>
                            new Chart(document.getElementById('chart'), {{
                                type: 'bar',
                                data: {json.dumps(result['visualization'])},
                                options: {{ 
                                    responsive: true, 
                                    scales: {{ y: {{ beginAtZero: true, max: 1 }} }},
                                    plugins: {{ legend: {{ display: false }} }}
                                }}
                            }});
                        </script>
                        """,
                        height=300
                    )
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            st.error(f"Error analyzing text: {str(e)}")

# Batch CSV analysis
st.header("Batch Analyze CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv", help="Upload a CSV with text data")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        
        # Column selection
        st.subheader("Select Columns")
        col1, col2 = st.columns(2)
        with col1:
            text_columns = [col for col in columns if any(term in col.lower() for term in ['text', 'comment', 'review'])] or columns
            text_column = st.selectbox("Text column", text_columns, help="Column containing text to analyze")
        with col2:
            date_columns = [col for col in columns if any(term in col.lower() for term in ['date', 'time'])] or columns
            date_column = st.selectbox("Date column (optional)", ["None"] + date_columns, help="Column with dates for trend analysis")
            date_column = None if date_column == "None" else date_column

        if st.button("Analyze CSV", key="analyze_csv"):
            uploaded_file.seek(0)  # Reset file pointer
            with st.spinner("Analyzing CSV..."):
                results = analyzer.batch_analyze(uploaded_file, text_column, date_column)
            st.subheader("Batch Analysis Results")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.json(results)
            with col2:
                st.subheader("Sentiment Distribution")
                components.html(
                    f"""
                    <canvas id="agg-chart"></canvas>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                    <script>
                        new Chart(document.getElementById('agg-chart'), {{
                            type: 'pie',
                            data: {json.dumps(results['aggregate_visualization'])},
                            options: {{ 
                                responsive: true,
                                plugins: {{ legend: {{ position: 'right' }} }}
                            }}
                        }});
                    </script>
                    """,
                    height=300
                )
            # Time trend chart
            if results.get('time_trend_data'):
                st.subheader("Sentiment Trends Over Time")
                components.html(
                    f"""
                    <canvas id="trend-chart"></canvas>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                    <script>
                        new Chart(document.getElementById('trend-chart'), {{
                            type: 'line',
                            data: {json.dumps(results['time_trend_data']['score_trends'])},
                            options: {{ 
                                responsive: true,
                                scales: {{ y: {{ beginAtZero: true, max: 1 }} }}
                            }}
                        }});
                    </script>
                    """,
                    height=400
                )
    except Exception as e:
        logger.error(f"Error batch analyzing: {str(e)}")
        st.error("Error: Please upload a valid CSV file")

# Feedback form
st.header("Submit Feedback")
with st.form("feedback"):
    feedback_text = st.text_input("Feedback text", placeholder="Enter text you analyzed")
    sentiment = st.selectbox("Correct sentiment", ["Positive", "Neutral", "Negative"], help="What should the sentiment be?")
    submit_feedback = st.form_submit_button("Submit")
    if submit_feedback:
        try:
            if not feedback_text or not sentiment:
                st.error("Please provide both text and sentiment")
            else:
                success = analyzer.store_feedback(feedback_text, sentiment)
                if success:
                    st.success("Feedback submitted successfully")
                else:
                    st.error("Error storing feedback")
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            st.error(f"Error storing feedback: {str(e)}")