import os
import numpy as np
import pandas as pd
import logging
import re
import random
from string import punctuation
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import TensorFlow, but provide a fallback
try:
    import tensorflow as tf
    tf_available = True
    logger.info("TensorFlow is available")
except ImportError:
    tf_available = False
    logger.warning("TensorFlow is not available, using fallback classifier")

class SentimentAnalyzer:
    """Class for sentiment analysis using a pre-trained model"""
    
    def __init__(self, model_path=None):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path (str, optional): Path to pre-trained model. 
                                        Defaults to None, which will use a default path.
        """
        self.model = None
        self.model_loaded = False
        
        # Comprehensive stopwords list
        self.stopwords = {
            # English stopwords
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'for', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'to', 'from', 'in', 'out', 'on', 'off', 'over',
            'under', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'up', 'down', 'of', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'their', 'his', 'her', 'its', 'our', 'your', 'my', 'mine', 'yours', 'ours',
            'theirs', 'me', 'him', 'us', 'them', 'would', 'should', 'could', 'can',
            'will', 'shall', 'may', 'might', 'must', 'there', 'here', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some',
            'other', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'once', 'now', 'also', 'again', 'always', 'often',
            'never', 'ever', 'while', 'still', 'yet', 'even', 'rather', 'quite',
            'enough', 'many', 'much', 'several', 'however', 'moreover', 'therefore',
            'hence', 'though', 'although', 'despite', 'since', 'until', 'unless',
            'whereas', 'whether', 'whatever', 'whoever', 'whenever', 'wherever',
            'whichever', 'somehow', 'sometimes', 'somewhere', 'anyway', 'anyhow',
            'anywhere', 'anyone', 'anything', 'anyway', 'anyplace', 'anytime',
            'already', 'almost', 'nearly', 'finally', 'eventually', 'possibly',
            'probably', 'actually', 'basically', 'generally', 'literally', 'really',
            'surely', 'certainly', 'obviously', 'naturally', 'apparently', 'clearly',
            'simply', 'namely', 'specifically', 'especially', 'particularly', 'notably',
            'chiefly', 'mainly', 'mostly', 'largely', 'primarily', 'principally',
            'significantly', 'substantially', 'essentially', 'consequently', 'accordingly',
            'thus', 'thereby', 'hereby', 'thereof', 'therein', 'thereupon', 'thereafter',
            'thereto', 'therewith', 'therefor', 'therefrom', 'thereabouts', 'thereafter',
            'whereof', 'whereupon', 'whereby', 'wherein', 'wherewith', 'wherefore',
            'whereafter', 'whereas', 'whither', 'hither', 'thither', 'whence', 'thence',
            
            # Common filler words
            'like', 'um', 'uh', 'er', 'ah', 'mm', 'hmm', 'huh', 'oh', 'eh', 'well',
            'yeah', 'yes', 'no', 'okay', 'ok', 'right', 'mhm', 'actually', 'basically',
            'literally', 'seriously', 'totally', 'absolutely', 'definitely', 'certainly',
            'surely', 'probably', 'possibly',
            
            # Common contractions
            "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't",
            "shouldn't", "mightn't", "mustn't", "needn't", "shan't", "im", "ive",
            "id", "youre", "youve", "youll", "youve", "youre", "hes", "shes", "its",
            "were", "theyll", "theyve", "theyre", "weve", "well", "whats", "thats",
            
            # Common single letters that may appear
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            
            # Common numbers and symbols as words
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            'hundred', 'thousand', 'million', 'billion', 'trillion',
            
            # Common units
            'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
            'time', 'times', 'hour', 'hours', 'minute', 'minutes', 'second', 'seconds',
            
            # Web-specific common words
            'http', 'https', 'www', 'com', 'org', 'net', 'html', 'url', 'website',
            'click', 'link', 'site', 'web', 'page', 'pages', 'post', 'posts', 'comment',
            'comments', 'user', 'users', 'email', 'username', 'login', 'logout', 'password',
            
            # Business and product review common words
            'product', 'products', 'service', 'services', 'customer', 'customers',
            'review', 'reviews', 'star', 'stars', 'rating', 'ratings', 'rate', 'rates',
            'buy', 'buying', 'bought', 'purchase', 'purchased', 'purchasing', 'order',
            'ordered', 'ordering', 'orders', 'item', 'items', 'shipping', 'shipped',
            'delivery', 'delivered', 'delivering', 'package', 'packaging', 'return',
            'returned', 'returning', 'refund', 'refunded', 'refunding', 'price', 'prices',
            'priced', 'cost', 'costs', 'costing', 'charge', 'charged', 'charging', 'fee',
            'fees', 'pay', 'paid', 'paying', 'payment', 'payments'
        }
        
        # Add emoji mappings for emotions
        self.emotion_emojis = {
            'Happy': '😊',      # Smiling face with smiling eyes
            'Satisfied': '😌',  # Relieved face
            'Neutral': '😐',    # Neutral face
            'Sad': '😔',        # Pensive face
            'Angry': '😡'       # Pouting face
        }
        
        # Try to load the model if TensorFlow is available
        if tf_available and model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.model = DummySentimentClassifier()
                logger.info("Using fallback classifier")
        else:
            self.model = DummySentimentClassifier()
            logger.info("Using fallback classifier")
    
    def analyze_text(self, text):
        """
        Analyze text for sentiment
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        if not text or len(text.strip()) == 0:
            return {
                'text': text,
                'sentiment': 'Neutral',
                'confidence': 1.0,
                'scores': {'Positive': 0.0, 'Neutral': 1.0, 'Negative': 0.0},
                'visualization': self._prepare_viz_data([0.0, 1.0, 0.0]),
                'aspects': [],
                'emotions': [],
                'keywords': []
            }
        
        # Get sentiment scores
        scores = self._get_sentiment_scores(text)
        
        # Determine sentiment class
        sentiment = self._get_sentiment_class(scores)
        
        # Calculate confidence
        confidence = float(max(scores))
        
        # Extract aspects
        aspects = self._extract_aspects(text, sentiment)
        
        # Detect emotions
        emotions = self._detect_emotions(text, scores)
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        # Prepare visualization data
        viz_data = self._prepare_viz_data(scores)
        
        # Return results
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {
                'Positive': float(scores[0]),
                'Neutral': float(scores[1]),
                'Negative': float(scores[2])
            },
            'visualization': viz_data,
            'aspects': aspects,
            'emotions': emotions,
            'keywords': keywords
        }
    
    def _get_sentiment_scores(self, text):
        """
        Get sentiment scores from the model
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: List of probability scores for each sentiment class
        """
        try:
            if self.model_loaded:
                # Preprocess text
                # This is a placeholder - in a real implementation, you would use
                # the same preprocessing as during training
                processed_text = self._preprocess_text(text)
                
                # Get predictions from model
                predictions = self.model.predict([processed_text])
                
                # Convert ndarray to list and return scores
                return [float(score) for score in predictions[0]]
            else:
                # Use fallback classifier
                scores = self.model.predict(text)
                # Convert numpy values to Python native types
                return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Error getting sentiment scores: {str(e)}")
            # Return default scores
            return [0.1, 0.8, 0.1]  # [positive, neutral, negative]
    
    def _preprocess_text(self, text):
        """
        Preprocess text for model input
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # This is a placeholder - in a real implementation, you would use
        # the same preprocessing as during training
        return text.lower()
    
    def _get_sentiment_class(self, scores):
        """
        Get the sentiment class based on scores and threshold
        
        Args:
            scores (list): List of probability scores
            
        Returns:
            str: Sentiment class
        """
        # Get index of max score
        max_index = np.argmax(scores)
        
        # Map to sentiment class
        sentiments = ['Positive', 'Neutral', 'Negative']
        return sentiments[max_index]
    
    def _prepare_viz_data(self, scores):
        """
        Prepare data for visualization
        
        Args:
            scores (list): List of probability scores
            
        Returns:
            dict: Data formatted for chart.js
        """
        # Ensure scores are Python native types
        scores = [float(score) for score in scores]
        
        # Format data for chart.js
        return {
            'labels': ['Positive', 'Neutral', 'Negative'],
            'datasets': [{
                'label': 'Sentiment Scores',
                'data': scores,
                'backgroundColor': [
                    'rgba(75, 192, 192, 0.5)',  # Positive
                    'rgba(54, 162, 235, 0.5)',  # Neutral
                    'rgba(255, 99, 132, 0.5)'   # Negative
                ],
                'borderColor': [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                'borderWidth': 1
            }]
        }
    
    def _extract_aspects(self, text, overall_sentiment):
        """
        Extract aspects and their sentiments
        
        This is a simplified implementation that would need to be enhanced
        with a proper aspect-based sentiment analysis model
        
        Args:
            text (str): The text to analyze
            overall_sentiment (str): The overall sentiment
            
        Returns:
            list: List of aspects and their sentiments
        """
        # Simple implementation - extract noun phrases
        aspects = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]', text)
        
        # Look for potential aspect phrases
        aspect_patterns = [
            r'(?:the|this|that|these|those)\s+(\w+)',
            r'(?:good|great|excellent|amazing|terrible|horrible|poor|bad)\s+(\w+)',
            r'(\w+)\s+(?:is|was|are|were)'
        ]
        
        # Extract aspects
        extracted = set()
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Check for aspects using patterns
            for pattern in aspect_patterns:
                matches = re.finditer(pattern, sentence.lower())
                for match in matches:
                    aspect = match.group(1)
                    if len(aspect) > 3 and aspect.isalpha() and aspect not in extracted and aspect not in self.stopwords:
                        # Determine aspect sentiment (simplified)
                        sentiment = overall_sentiment
                        
                        # If the aspect is in a negative context, flip the sentiment
                        neg_words = ['not', "n't", 'never', 'no', 'nothing', 'neither', 'nor']
                        neg_context = any(neg in sentence.lower() for neg in neg_words)
                        
                        if neg_context:
                            if sentiment == 'Positive':
                                sentiment = 'Negative'
                            elif sentiment == 'Negative':
                                sentiment = 'Positive'
                        
                        aspects.append({
                            'aspect': aspect.capitalize(),
                            'sentiment': sentiment
                        })
                        extracted.add(aspect)
                        
                        # Limit the number of aspects
                        if len(aspects) >= 5:
                            return aspects
        
        return aspects
    
    def _detect_emotions(self, text, scores):
        """
        Detect emotions in text
        
        This is a simplified implementation that would need to be enhanced
        with a proper emotion detection model
        
        Args:
            text (str): The text to analyze
            scores (list): Sentiment scores
            
        Returns:
            list: List of emotions and their intensities
        """
        # Simple implementation - map to emotions based on scores and keywords
        emotions = []
        text = text.lower()
        
        # Emotion keywords
        emotion_keywords = {
            'Happy': ['happy', 'joy', 'excellent', 'great', 'love', 'enjoy', 'awesome', 'fun'],
            'Satisfied': ['satisfied', 'good', 'pleased', 'content', 'fine', 'ok', 'okay'],
            'Neutral': ['neutral', 'average', 'mediocre', 'middle', 'standard'],
            'Sad': ['sad', 'unhappy', 'disappointed', 'upset', 'regret', 'sorry'],
            'Angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'terrible']
        }
        
        # Check for emotion keywords
        emotion_scores = {
            'Happy': 0.1,
            'Satisfied': 0.1,
            'Neutral': 0.1,
            'Sad': 0.1,
            'Angry': 0.1
        }
        
        # Base emotion on sentiment scores
        positive_score = float(scores[0])
        neutral_score = float(scores[1])
        negative_score = float(scores[2])
        
        emotion_scores['Happy'] += positive_score * 0.7
        emotion_scores['Satisfied'] += positive_score * 0.3
        emotion_scores['Neutral'] += neutral_score
        emotion_scores['Sad'] += negative_score * 0.6
        emotion_scores['Angry'] += negative_score * 0.4
        
        # Adjust based on keywords
        words = re.findall(r'\b\w+\b', text)
        for emotion, keywords in emotion_keywords.items():
            for word in words:
                if word in keywords:
                    emotion_scores[emotion] += 0.2
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
                emotion_scores[emotion] = min(emotion_scores[emotion], 1.0)
        
        # Create emotion list
        for emotion, score in emotion_scores.items():
            if score > 0.2:  # Only include emotions with significant scores
                emotions.append({
                    'emotion': emotion,
                    'emoji': self.emotion_emojis.get(emotion, ''),
                    'intensity': float(score)
                })
        
        # Sort by intensity
        emotions.sort(key=lambda x: x['intensity'], reverse=True)
        
        return emotions[:3]  # Return top 3 emotions
    
    def _extract_keywords(self, text):
        """
        Extract important keywords from text
        
        This is a simplified implementation that would need to be enhanced
        with proper NLP techniques
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: List of keywords and their importance
        """
        # Simple implementation - extract most frequent words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords and short words
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 3 and word.isalpha()]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Convert to list of tuples and sort
        word_list = [(word, count) for word, count in word_counts.items()]
        word_list.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to keyword objects
        keywords = []
        max_count = word_list[0][1] if word_list else 1
        
        for word, count in word_list[:10]:  # Get top 10 keywords
            importance = float(count / max_count)
            keywords.append({
                'word': word,
                'importance': importance
            })
        
        return keywords
        
    def batch_analyze(self, file, text_column=None, date_column=None):
        """
        Analyze multiple texts from a CSV file
        
        Args:
            file: CSV file with text samples
            text_column: Name of the column containing text to analyze
            date_column: Name of the column containing dates for time trend analysis
            
        Returns:
            dict: Dictionary with batch analysis results
        """
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Check if text_column is provided and exists
            if text_column and text_column in df.columns:
                pass
            # Check if 'text' column exists
            elif 'text' in df.columns:
                text_column = 'text'
            # Try to use the first column
            else:
                text_column = df.columns[0]
            
            logger.debug(f"Using column '{text_column}' for batch analysis")
            
            # Analyze each text
            results = []
            for idx, row in df.iterrows():
                text = str(row[text_column])
                if text and len(text.strip()) > 0:
                    analysis = self.analyze_text(text)
                    
                    # Add date if available
                    if date_column and date_column in df.columns:
                        try:
                            date_value = row[date_column]
                            if isinstance(date_value, str):
                                analysis['date'] = date_value
                            else:
                                analysis['date'] = str(date_value)
                        except:
                            pass
                    
                    results.append(analysis)
                
                # Progress logging for large files
                if idx > 0 and idx % 100 == 0:
                    logger.debug(f"Processed {idx} rows...")
            
            # Basic sentiment counts
            sentiment_counts = {
                'Positive': sum(1 for r in results if r['sentiment'] == 'Positive'),
                'Neutral': sum(1 for r in results if r['sentiment'] == 'Neutral'),
                'Negative': sum(1 for r in results if r['sentiment'] == 'Negative')
            }
            
            # Prepare aggregate visualization data (pie chart)
            aggregate_viz = {
                'labels': list(sentiment_counts.keys()),
                'datasets': [{
                    'label': 'Sentiment Distribution',
                    'data': list(sentiment_counts.values()),
                    'backgroundColor': [
                        'rgba(75, 192, 192, 0.5)',  # Positive
                        'rgba(54, 162, 235, 0.5)',  # Neutral
                        'rgba(255, 99, 132, 0.5)'   # Negative
                    ],
                    'borderColor': [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    'borderWidth': 1
                }]
            }
            
            # Additional visualizations
            word_cloud_data = self._aggregate_keywords(results)
            emotion_data = self._aggregate_emotions(results)
            score_distribution = self._calculate_score_distribution(results)
            top_reviews = self._extract_top_reviews(results)
            
            # Time trend analysis if date column is provided
            time_trend_data = None
            if date_column and any('date' in r for r in results):
                time_trend_data = self._analyze_time_trends(results)
            
            # Return enhanced batch analysis results
            return {
                'sample_count': len(results),
                'sentiment_counts': sentiment_counts,
                'aggregate_visualization': aggregate_viz,
                'detailed_results': results,
                'word_cloud_data': word_cloud_data,
                'emotion_data': emotion_data,
                'score_distribution': score_distribution,
                'top_reviews': top_reviews,
                'time_trend_data': time_trend_data
            }
            
        except Exception as e:
            logger.error(f"Error batch analyzing: {str(e)}")
            raise
    
    def _analyze_time_trends(self, results):
        """Analyze sentiment trends over time"""
        try:
            # Filter results with date
            date_results = [r for r in results if 'date' in r]
            
            if not date_results:
                return None
                
            # Convert dates to datetime objects
            for result in date_results:
                try:
                    # Try to parse date
                    result['datetime'] = pd.to_datetime(result['date'])
                except:
                    # If parsing fails, set to None
                    result['datetime'] = None
            
            # Filter out results with invalid dates
            date_results = [r for r in date_results if r['datetime'] is not None]
            
            if not date_results:
                return None
                
            # Sort by date
            date_results.sort(key=lambda x: x['datetime'])
            
            # Determine appropriate time grouping based on date range
            min_date = min(r['datetime'] for r in date_results)
            max_date = max(r['datetime'] for r in date_results)
            date_range = (max_date - min_date).days
            
            # Define grouping
            if date_range <= 30:
                # Group by day
                group_format = '%Y-%m-%d'
                group_label = 'day'
            elif date_range <= 365:
                # Group by week
                group_format = '%Y-%W'  # Year and week number
                group_label = 'week'
            else:
                # Group by month
                group_format = '%Y-%m'
                group_label = 'month'
            
            # Group results by time period
            time_groups = {}
            for result in date_results:
                period = result['datetime'].strftime(group_format)
                if period not in time_groups:
                    time_groups[period] = {
                        'period': period,
                        'period_date': result['datetime'],
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0,
                        'total': 0,
                        'positive_score': 0,
                        'negative_score': 0,
                        'neutral_score': 0,
                        'avg_positive': 0,
                        'avg_negative': 0,
                        'avg_neutral': 0
                    }
                
                # Increment sentiment counts
                sentiment = result['sentiment']
                time_groups[period][sentiment.lower()] += 1
                time_groups[period]['total'] += 1
                
                # Add scores
                time_groups[period]['positive_score'] += result['scores']['Positive']
                time_groups[period]['negative_score'] += result['scores']['Negative']
                time_groups[period]['neutral_score'] += result['scores']['Neutral']
            
            # Calculate averages
            for period, data in time_groups.items():
                if data['total'] > 0:
                    data['avg_positive'] = data['positive_score'] / data['total']
                    data['avg_negative'] = data['negative_score'] / data['total']
                    data['avg_neutral'] = data['neutral_score'] / data['total']
            
            # Convert to list and sort by date
            time_series = list(time_groups.values())
            time_series.sort(key=lambda x: x['period_date'])
            
            # Format for chart.js
            chart_data = {
                'labels': [ts['period'] for ts in time_series],
                'datasets': [
                    {
                        'label': 'Positive',
                        'data': [float(ts['avg_positive']) for ts in time_series],
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'fill': False
                    },
                    {
                        'label': 'Neutral',
                        'data': [float(ts['avg_neutral']) for ts in time_series],
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                        'fill': False
                    },
                    {
                        'label': 'Negative',
                        'data': [float(ts['avg_negative']) for ts in time_series],
                        'borderColor': 'rgba(255, 99, 132, 1)',
                        'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                        'fill': False
                    }
                ],
                'group_by': group_label
            }
            
            # Count percentage chart
            count_data = {
                'labels': [ts['period'] for ts in time_series],
                'datasets': [
                    {
                        'label': 'Positive',
                        'data': [round(ts['positive'] / ts['total'] * 100, 1) if ts['total'] > 0 else 0 for ts in time_series],
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'fill': False
                    },
                    {
                        'label': 'Neutral',
                        'data': [round(ts['neutral'] / ts['total'] * 100, 1) if ts['total'] > 0 else 0 for ts in time_series],
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                        'fill': False
                    },
                    {
                        'label': 'Negative',
                        'data': [round(ts['negative'] / ts['total'] * 100, 1) if ts['total'] > 0 else 0 for ts in time_series],
                        'borderColor': 'rgba(255, 99, 132, 1)',
                        'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                        'fill': False
                    }
                ],
                'group_by': group_label
            }
            
            return {
                'score_trends': chart_data,
                'percentage_trends': count_data,
                'time_series': time_series
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time trends: {str(e)}")
            return None
    
    def _aggregate_keywords(self, results):
        """Aggregate keywords across all analyzed texts"""
        # Collect all keywords
        all_keywords = {}
        
        for result in results:
            for keyword in result['keywords']:
                word = keyword['word']
                importance = keyword['importance']
                
                if word in all_keywords:
                    all_keywords[word] += importance
                else:
                    all_keywords[word] = importance
        
        # Convert to list and sort
        aggregated_keywords = [{'word': word, 'importance': float(importance)} 
                              for word, importance in all_keywords.items()]
        aggregated_keywords.sort(key=lambda x: x['importance'], reverse=True)
        
        # Return top 50 keywords
        return aggregated_keywords[:50]
    
    def _aggregate_emotions(self, results):
        """Aggregate emotions across all analyzed texts"""
        # Track emotions and their intensities
        all_emotions = {}
        emotion_counts = {}
        
        for result in results:
            for emotion in result['emotions']:
                name = emotion['emotion']
                intensity = emotion['intensity']
                
                if name in all_emotions:
                    all_emotions[name] += intensity
                    emotion_counts[name] += 1
                else:
                    all_emotions[name] = intensity
                    emotion_counts[name] = 1
        
        # Calculate average intensity for each emotion
        emotion_data = {
            'labels': [],
            'datasets': [{
                'label': 'Emotion Intensity',
                'data': [],
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'borderColor': 'rgba(75, 192, 192, 1)',
                'pointBackgroundColor': 'rgba(75, 192, 192, 1)',
                'pointBorderColor': '#fff'
            }]
        }
        
        # Common emotions to show even if not detected
        common_emotions = ['Happy', 'Satisfied', 'Neutral', 'Sad', 'Angry']
        
        # Ensure common emotions are included
        for emotion in common_emotions:
            if emotion not in all_emotions:
                all_emotions[emotion] = 0
                emotion_counts[emotion] = 1
        
        # Calculate averages and prepare chart data
        for emotion, total in all_emotions.items():
            average = float(total / emotion_counts[emotion])
            emotion_data['labels'].append(emotion + ' ' + self.emotion_emojis.get(emotion, ''))
            emotion_data['datasets'][0]['data'].append(round(average, 2))
        
        return emotion_data
    
    def _calculate_score_distribution(self, results):
        """Calculate the distribution of sentiment scores"""
        # Create bins for the histogram
        bins = 10
        bin_size = 1.0 / bins
        
        # Initialize bins
        positive_bins = [0] * bins
        negative_bins = [0] * bins
        neutral_bins = [0] * bins
        
        # Populate bins
        for result in results:
            pos_score = result['scores']['Positive']
            neg_score = result['scores']['Negative']
            neut_score = result['scores']['Neutral']
            
            # Determine which bin each score belongs to
            pos_bin = min(int(pos_score / bin_size), bins - 1)
            neg_bin = min(int(neg_score / bin_size), bins - 1)
            neut_bin = min(int(neut_score / bin_size), bins - 1)
            
            # Increment bins
            positive_bins[pos_bin] += 1
            negative_bins[neg_bin] += 1
            neutral_bins[neut_bin] += 1
        
        # Create bin labels (0-10%, 10-20%, etc.)
        labels = [f"{int(i*bin_size*100)}-{int((i+1)*bin_size*100)}%" for i in range(bins)]
        
        # Return histogram data
        return {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Positive Scores',
                    'data': positive_bins,
                    'backgroundColor': 'rgba(75, 192, 192, 0.5)'
                },
                {
                    'label': 'Neutral Scores',
                    'data': neutral_bins,
                    'backgroundColor': 'rgba(54, 162, 235, 0.5)'
                },
                {
                    'label': 'Negative Scores',
                    'data': negative_bins,
                    'backgroundColor': 'rgba(255, 99, 132, 0.5)'
                }
            ]
        }
    
    def _extract_top_reviews(self, results):
        """Extract the most positive and most negative reviews"""
        # Sort by confidence and sentiment
        positive_reviews = [r for r in results if r['sentiment'] == 'Positive']
        negative_reviews = [r for r in results if r['sentiment'] == 'Negative']
        
        # Sort by confidence score
        positive_reviews.sort(key=lambda x: x['confidence'], reverse=True)
        negative_reviews.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get top 5 of each (or fewer if not enough examples)
        top_positive = positive_reviews[:min(5, len(positive_reviews))]
        top_negative = negative_reviews[:min(5, len(negative_reviews))]
        
        return {
            'positive': top_positive,
            'negative': top_negative
        }
        
    def store_feedback(self, text, correct_sentiment):
        """
        Store user feedback for model improvement
        
        Args:
            text (str): The text that was analyzed
            correct_sentiment (str): The correct sentiment according to user
            
        Returns:
            bool: True if feedback was stored successfully
        """
        try:
            # In a real implementation, you would store this feedback in a database
            # for later use in model improvement
            
            logger.info(f"Received feedback: '{text}' should be '{correct_sentiment}'")
            
            # For now, just return success
            return True
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            return False


class DummySentimentClassifier:
    """
    Dummy sentiment classifier for fallback when model can't be loaded
    Uses simple rule-based approach for demo purposes
    """
    
    def __init__(self):
        # Define positive and negative word lists
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'terrific', 'outstanding', 'brilliant', 'superb', 'awesome',
            'best', 'better', 'love', 'happy', 'pleased', 'satisfied',
            'like', 'enjoy', 'enjoyed', 'impressive', 'beautiful',
            'perfect', 'positive', 'nice', 'helpful', 'recommended',
            'quality', 'valuable', 'superior', 'worth', 'convenient',
            'smooth', 'easy', 'improvement', 'improved', 'effective',
            'efficient', 'recommend', 'glad', 'delighted', 'joy',
            'success', 'successful', 'exceptional', 'favorable', 'benefit'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'horrible', 'awful', 'poor', 'disappointing',
            'disappointed', 'disappoints', 'worst', 'waste', 'useless',
            'hate', 'dislike', 'problem', 'issue', 'difficult', 'difficult',
            'negative', 'ugly', 'annoying', 'annoyed', 'annoys', 'slow',
            'expensive', 'overpriced', 'inconvenient', 'unhappy', 'sad',
            'angry', 'frustrated', 'frustrating', 'mediocre', 'failure',
            'failed', 'fails', 'broken', 'bug', 'inferior', 'cheap',
            'unreliable', 'unrecommended', 'mistake', 'regret', 'avoid',
            'complaint', 'complained', 'complaining', 'faulty', 'defect'
        }
        
        # Define negation words
        self.negations = {
            'not', 'no', 'never', 'none', 'neither', 'nor', "n't", 'without'
        }
    
    def predict(self, text):
        """
        Predict sentiment using word lists
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: Sentiment scores [positive, neutral, negative]
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into words and remove punctuation
        words = re.findall(r'\b\w+\b', text)
        
        # Count positive and negative words
        pos_count = 0
        neg_count = 0
        
        # Keep track of negation
        negated = False
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.negations or (len(word) > 3 and word.endswith("n't")):
                negated = True
                continue
            
            # Check sentiment
            if word in self.positive_words:
                if negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in self.negative_words:
                if negated:
                    pos_count += 1
                else:
                    neg_count += 1
            
            # Reset negation (negation typically affects only the next sentiment word)
            if word not in self.negations and not (len(word) > 3 and word.endswith("n't")):
                negated = False
        
        # Calculate sentiment scores
        total = pos_count + neg_count
        
        if total == 0:
            # No sentiment words found, return neutral
            return np.array([0.1, 0.8, 0.1])
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        
        # Add randomness for demo variety
        pos_score = min(1.0, max(0.0, pos_score + random.uniform(-0.1, 0.1)))
        neg_score = min(1.0, max(0.0, neg_score + random.uniform(-0.1, 0.1)))
        
        # Normalize
        total = pos_score + neg_score
        if total > 0:
            pos_score = pos_score / total
            neg_score = neg_score / total
        else:
            pos_score = 0.5
            neg_score = 0.5
        
        # Calculate neutral score
        if abs(pos_score - neg_score) < 0.3:
            # Scores are close, increase neutral score
            neut_score = random.uniform(0.4, 0.6)
            # Adjust positive and negative accordingly
            pos_score = pos_score * (1 - neut_score)
            neg_score = neg_score * (1 - neut_score)
        else:
            # Scores have clear winner, lower neutral score
            neut_score = random.uniform(0.1, 0.3)
            # Adjust positive and negative
            factor = (1 - neut_score) / (pos_score + neg_score)
            pos_score = pos_score * factor
            neg_score = neg_score * factor
        
        # Return scores as regular Python floats, not numpy types
        return np.array([float(pos_score), float(neut_score), float(neg_score)])