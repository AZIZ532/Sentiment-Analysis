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

# App title
st.title("Sentiment Analysis App")

# Single text analysis
st.header("Analyze Text")
text = st.text_area("Enter text for sentiment analysis")
if st.button("Analyze"):
    try:
        if not text:
            st.error("No text provided")
        else:
            result = analyzer.analyze_text(text)
            st.json(result)
            # Render sentiment bar chart
            components.html(
                f"""
                <canvas id="chart"></canvas>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    new Chart(document.getElementById('chart'), {{
                        type: 'bar',
                        data: {json.dumps(result['visualization'])},
                        options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
                    }});
                </script>
                """,
                height=400
            )
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        st.error(f"Error analyzing text: {str(e)}")

# Batch CSV analysis
st.header("Batch Analyze CSV")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        
        # Column selection
        st.subheader("Select Columns")
        text_columns = [col for col in columns if any(term in col.lower() for term in ['text', 'comment', 'review'])]
        date_columns = [col for col in columns if any(term in col.lower() for term in ['date', 'time'])]
        
        text_column = st.selectbox("Select text column", text_columns if text_columns else columns)
        date_column = st.selectbox("Select date column (optional)", ["None"] + (date_columns if date_columns else columns))
        date_column = None if date_column == "None" else date_column

        if st.button("Analyze CSV"):
            uploaded_file.seek(0)  # Reset file pointer
            results = analyzer.batch_analyze(uploaded_file, text_column, date_column)
            st.json(results)
            # Render aggregate pie chart
            components.html(
                f"""
                <canvas id="agg-chart"></canvas>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    new Chart(document.getElementById('agg-chart'), {{
                        type: 'pie',
                        data: {json.dumps(results['aggregate_visualization'])},
                        options: {{ responsive: true }}
                    }});
                </script>
                """,
                height=400
            )
            # Render time trend chart if available
            if results.get('time_trend_data'):
                components.html(
                    f"""
                    <canvas id="trend-chart"></canvas>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <script>
                        new Chart(document.getElementById('trend-chart'), {{
                            type: 'line',
                            data: {json.dumps(results['time_trend_data']['score_trends'])},
                            options: {{ responsive: true }}
                        }});
                    </script>
                    """,
                    height=400
                )
    except Exception as e:
        logger.error(f"Error batch analyzing: {str(e)}")
        st.error(f"Error batch analyzing: {str(e)}")

# Feedback form
st.header("Submit Feedback")
with st.form("feedback"):
    feedback_text = st.text_input("Feedback text")
    sentiment = st.selectbox("Correct sentiment", ["Positive", "Neutral", "Negative"])
    submitted = st.form_submit_button("Submit")
    if submitted:
        try:
            if not feedback_text or not sentiment:
                st.error("Invalid feedback data")
            else:
                success = analyzer.store_feedback(feedback_text, sentiment)
                if success:
                    st.success("Feedback submitted successfully")
                else:
                    st.error("Error storing feedback")
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            st.error(f"Error storing feedback: {str(e)}")