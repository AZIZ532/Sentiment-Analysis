from flask import Flask, render_template, request, jsonify, abort
import os
import logging
from sentiment_model import SentimentAnalyzer
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text for sentiment"""
    try:
        text = request.form.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'})
        
        # Analyze text
        results = analyzer.analyze_text(text)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': f'Error analyzing text: {str(e)}'})

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple text samples for sentiment"""
    try:
        file = request.files.get('file')
        
        if not file:
            return jsonify({'error': 'No file provided'})
        
        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})
        
        # If column selection is provided
        text_column = request.form.get('text_column', None)
        date_column = request.form.get('date_column', None)
        
        # Analyze file
        results = analyzer.batch_analyze(file, text_column, date_column)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error batch analyzing: {str(e)}")
        return jsonify({'error': f'Error batch analyzing: {str(e)}'})

@app.route('/get-csv-columns', methods=['POST'])
def get_csv_columns():
    """Get column names from CSV file"""
    try:
        file = request.files.get('file')
        
        if not file:
            return jsonify({'error': 'No file provided'})
        
        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Get column names
        columns = df.columns.tolist()
        
        # Detect potential date columns
        date_columns = []
        for col in columns:
            # Simple heuristic - check column name
            if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                date_columns.append(col)
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col].iloc[0:5])
                if col not in date_columns:
                    date_columns.append(col)
            except:
                pass
        
        # Detect potential text columns
        text_columns = []
        for col in columns:
            # Simple heuristic - check column name
            if any(text_term in col.lower() for text_term in ['text', 'comment', 'review', 'feedback', 'content', 'message', 'description']):
                text_columns.append(col)
            # Check data type - string
            if df[col].dtype == 'object':
                # Check average string length
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 20 and col not in text_columns:
                    text_columns.append(col)
        
        return jsonify({
            'columns': columns,
            'potential_date_columns': date_columns,
            'potential_text_columns': text_columns
        })
    except Exception as e:
        logger.error(f"Error getting CSV columns: {str(e)}")
        return jsonify({'error': f'Error getting CSV columns: {str(e)}'})

@app.route('/feedback', methods=['POST'])
def feedback():
    """Process user feedback for sentiment analysis"""
    try:
        data = request.json
        
        if not data or 'text' not in data or 'sentiment' not in data:
            return jsonify({'error': 'Invalid feedback data'})
        
        text = data['text']
        sentiment = data['sentiment']
        
        # Store feedback
        success = analyzer.store_feedback(text, sentiment)
        
        if success:
            return jsonify({'message': 'Feedback submitted successfully'})
        else:
            return jsonify({'error': 'Error storing feedback'})
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        return jsonify({'error': f'Error storing feedback: {str(e)}'})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
