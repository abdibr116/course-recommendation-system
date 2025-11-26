"""
Course Recommendation System - Flask Application

Author: Rana M Almuaied
Email: raanosh12@gmail.com
Date: November 26, 2025

Brief:
    Flask web application providing course recommendations using content-based
    filtering with TF-IDF vectorization, K-NN similarity, and user feedback
    collection for model evaluation.

Usage:
    Start the Flask server:
    >>> python app.py
    
    Then access the application:
    - Main interface: http://127.0.0.1:5000/
    - Statistics: http://127.0.0.1:5000/stats
    - Metrics: http://127.0.0.1:5000/metrics?k=10
"""

# Import required libraries for web framework and data processing
from flask import Flask, render_template, request, jsonify  # Flask web framework components
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library for array operations
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization for similarity matching
from sklearn.preprocessing import StandardScaler  # Feature scaling for normalization
from sklearn.neighbors import NearestNeighbors  # K-Nearest Neighbors algorithm for finding similar items
from scipy.sparse import hstack  # Horizontal stacking of sparse matrices
import sys  # System-specific parameters and functions
import os  # Operating system interface for file operations
import re  # Regular expressions for text cleaning
import json  # JSON handling for feedback data
from datetime import datetime  # Date and time operations

# Import NLTK for text preprocessing
import nltk
from nltk.stem import WordNetLemmatizer  # Lemmatization
from nltk.corpus import stopwords  # Stop words removal

# Initialize Flask application instance
app = Flask(__name__)

# Global variables for the recommendation system
df = None  # DataFrame to store combined course data
model = None  # K-Nearest Neighbors model for similarity search
vectorizer = None  # TF-IDF vectorizer for text feature extraction
X_combined = None  # Combined feature matrix (text + numeric)
X_num = None  # Numeric features matrix
scaler = None  # StandardScaler for normalizing numeric features
lemmatizer = None  # WordNet lemmatizer for text preprocessing
feedback_data = []  # Store user feedback for evaluation

def parse_number(x):
    """
    Convert human-readable number formats to floats.
    
    Brief:
        Parses enrollment numbers from various string formats (e.g., '5.3k', '1.2M')
        into standardized float values for numerical analysis.
    
    Usage:
        >>> parse_number('5.3k')
        5300.0
        >>> parse_number('1.2M')
        1200000.0
        >>> parse_number('12,345 students')
        12345.0
    
    Args:
        x: Input value (string or numeric) representing enrollment count
    
    Returns:
        float: Parsed numeric value or np.nan if input is null
    """
    # Check if value is missing/null
    if pd.isna(x):
        return np.nan

    # If already numeric, convert to float and return
    if not isinstance(x, str):
        return float(x)

    # Convert to lowercase and remove whitespace
    s = x.strip().lower()

    # Remove common words that might appear in enrollment numbers
    for word in ["students", "enrolled", "student", " ", "+"]:
        s = s.replace(word, "")

    # Remove comma separators (e.g., "12,345" -> "12345")
    s = s.replace(",", "")

    # Handle 'k' suffix (thousands)
    if s.endswith("k"):
        num = float(s[:-1]) * 1000
    # Handle 'm' suffix (millions)
    elif s.endswith("m"):
        num = float(s[:-1]) * 1_000_000
    # Handle plain numbers
    else:
        try:
            num = float(s)
        except:
            # Return 0 if conversion fails
            num = 0.0

    return num

def preprocess_text(text):
    """
    Preprocess text using tokenization, lemmatization, and stop word removal.
    
    Brief:
        Implements advanced text preprocessing as specified in the proposal:
        tokenization, lemmatization, and stop word removal to prepare text
        for vectorization and similarity analysis.
    
    Usage:
        >>> preprocess_text("Introduction to Machine Learning Algorithms")
        'introduction machine learning algorithm'
        
    Args:
        text (str): Raw text to preprocess
    
    Returns:
        str: Cleaned and preprocessed text
    """
    global lemmatizer
    
    # Handle null/empty values
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenization: split text into individual words (simple whitespace split)
    tokens = text.split()
    
    # Remove stop words (common words like 'the', 'is', 'at')
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization: reduce words to their root form
    # e.g., "running" -> "run", "better" -> "good"
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

def load_and_prepare_data():
    """
    Load course data from CSV files and prepare features for recommendation system.
    
    Brief:
        Initializes the course recommendation system by loading datasets from both
        Coursera and Udemy, normalizing schemas, extracting features, and training
        a K-Nearest Neighbors model for similarity-based recommendations.
    
    Usage:
        >>> load_and_prepare_data()
        Data loaded successfully! Total courses: 4569
        
    Side Effects:
        - Sets global variables: df, model, vectorizer, X_combined, X_num, scaler
        - Loads data from 'dataset/coursea_data.csv' and 'dataset/udemy_courses.csv'
        - Trains TF-IDF vectorizer and K-NN model on combined course data
    
    Returns:
        None
    """
    # Declare global variables to be modified
    global df, model, vectorizer, X_combined, X_num, scaler, lemmatizer
    
    # NLTK data is pre-downloaded in Docker image
    # No runtime download needed
    import nltk
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Load Coursera course data from CSV file
    df_coursera_raw = pd.read_csv("dataset/coursea_data.csv")
    
    # Normalize Coursera columns to match expected schema
    # New dataset has different column names: 'Course Title', 'Rating', 'Level', etc.
    df_coursera = pd.DataFrame({
        'course_title': df_coursera_raw['Course Title'],  # Course name
        'course_organization': df_coursera_raw['Offered By'],  # University/organization
        'course_Certificate_type': df_coursera_raw['Skill gain'],  # Skills/subject area
        'course_rating': df_coursera_raw['Rating'],  # Course rating
        'course_difficulty': df_coursera_raw['Level'],  # Difficulty level
        'course_students_enrolled': df_coursera_raw['Number of Review'],  # Use review count as proxy for enrollment
        'source': 'Coursera'  # Platform identifier
    })
    
    # Load Udemy course data from CSV file
    df_udemy = pd.read_csv("dataset/udemy_courses.csv")
    
    # Normalize Udemy columns to match Coursera structure
    # This creates a unified schema for both platforms
    df_udemy_normalized = pd.DataFrame({
        'course_title': df_udemy['course_title'],  # Course name
        'course_organization': 'Udemy',  # Set all Udemy courses to 'Udemy' organization
        'course_Certificate_type': df_udemy['subject'],  # Map subject to certificate type
        'course_rating': pd.to_numeric(df_udemy.get('course_rating', 0), errors='coerce'),  # Convert rating to numeric, default to 0
        'course_difficulty': df_udemy['level'],  # Difficulty level (Beginner, Intermediate, etc.)
        'course_students_enrolled': df_udemy['num_subscribers'],  # Map subscribers to enrollment count
        'source': 'Udemy'  # Add source identifier for filtering
    })
    
    # Combine both datasets into single DataFrame, reset index
    df = pd.concat([df_coursera, df_udemy_normalized], ignore_index=True)
    
    # Clean enrollment numbers using parse_number function
    # Converts formats like '5.3k' to 5300
    df["course_students_enrolled"] = df["course_students_enrolled"].apply(parse_number)
    
    # Clean course ratings, convert to numeric, handle errors gracefully
    df["course_rating"] = pd.to_numeric(df["course_rating"], errors="coerce")
    
    # Define numeric columns that will be used as features
    numeric_cols = ["course_rating", "course_students_enrolled"]
    
    # Fill any missing numeric values with 0.0
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    # Initialize StandardScaler for feature normalization
    scaler = StandardScaler()
    
    # Scale numeric features to have mean=0 and std=1
    # This ensures numeric features don't dominate the similarity calculation
    X_num = scaler.fit_transform(df[numeric_cols])
    
    # Apply advanced text preprocessing with tokenization and lemmatization
    # This implements the proposal's feature engineering requirements
    print("Preprocessing text data with tokenization and lemmatization...")
    df["text_preprocessed"] = df.apply(
        lambda row: preprocess_text(
            str(row["course_title"]) + " " +
            str(row["course_title"]) + " " +  # Double weight on title
            str(row["course_organization"]) + " " +
            str(row["course_Certificate_type"]) + " " +
            str(row["course_Certificate_type"]) + " " +  # Double weight on category
            str(row["course_difficulty"])
        ),
        axis=1
    )
    
    # Initialize TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit to 5000 most important features to reduce dimensionality
        ngram_range=(1, 2),  # Include both single words and two-word phrases
        min_df=1  # Include terms that appear in at least 1 document
    )
    
    # Transform preprocessed text data into TF-IDF feature matrix
    # This converts text to numerical vectors for similarity calculation
    X_text = vectorizer.fit_transform(df["text_preprocessed"])
    
    # Combine text features and numeric features horizontally
    # Creates a unified feature matrix for recommendation
    X_combined = hstack([X_text, X_num])
    
    # Convert to Compressed Sparse Row format for efficient operations
    X_combined = X_combined.tocsr()
    
    # Initialize K-Nearest Neighbors model
    model = NearestNeighbors(
        metric="cosine",  # Use cosine similarity for text-based matching
        algorithm="brute"  # Use brute-force search for accuracy
    )
    
    # Fit the model with our combined feature matrix
    model.fit(X_combined)
    
    # Print confirmation message with dataset size
    print(f"Data loaded successfully! Total courses: {len(df)}")

def find_similar_courses(query_text, top_n=10):
    """
    Find courses similar to a given text query with intelligent platform diversity.
    
    Brief:
        Performs content-based filtering using TF-IDF and K-NN to find courses
        matching the user's query. Implements smart diversity algorithm to balance
        results between Coursera and Udemy while filtering low-relevance matches.
    
    Usage:
        >>> results = find_similar_courses("Python programming", top_n=5)
        >>> print(results[['course_title', 'source', 'similarity_score']])
        
    Args:
        query_text (str): User's search query (e.g., "Python programming")
        top_n (int): Number of recommendations to return (default: 10)
    
    Returns:
        pandas.DataFrame: Top N similar courses with columns:
            - course_title, course_organization, course_Certificate_type
            - course_rating, course_difficulty, course_students_enrolled
            - source, similarity_score
    """
    # Declare global variables to be accessed
    global df, model, vectorizer, X_num
    
    # Preprocess user query using same text preprocessing pipeline
    preprocessed_query = preprocess_text(query_text)
    
    # Transform preprocessed query into TF-IDF vector using trained vectorizer
    query_vec_text = vectorizer.transform([preprocessed_query])
    
    # Create zero vector for numeric features (we don't have ratings/enrollment for queries)
    # Using zeros represents the mean after standardization
    zero_num = np.zeros((1, X_num.shape[1]))
    
    # Combine text and numeric features for query vector
    query_vec = hstack([query_vec_text, zero_num])
    
    # Get more candidates than needed to allow for filtering and diversity
    # Request 4x the desired results to have options for platform mixing
    n_candidates = min(top_n * 4, len(df))
    
    # Find nearest neighbors using cosine similarity
    # Returns distances and indices of most similar courses
    distances, indices = model.kneighbors(query_vec, n_neighbors=n_candidates)
    
    # Extract recommended courses from DataFrame using indices
    result = df.iloc[indices[0]].copy()
    
    # Store cosine distance for reference
    result["distance"] = distances[0]
    
    # Convert distance to similarity percentage (0-100%)
    # Cosine similarity = 1 - cosine distance
    result["similarity_score"] = (1 - distances[0]) * 100
    
    # Filter out low-relevance results (less than 10% match)
    # This prevents showing completely unrelated courses
    result = result[result['similarity_score'] > 10]
    
    # Separate results by platform for diversity analysis
    coursera_results = result[result['source'] == 'Coursera'].head(top_n)
    udemy_results = result[result['source'] == 'Udemy'].head(top_n)
    
    # Check if both platforms have sufficient relevant results
    # Only mix platforms if both have at least 2 relevant courses
    if len(coursera_results) >= 2 and len(udemy_results) >= 2:
        # Mix results from both platforms for diversity
        # Include 25% from Coursera (minimum 2) and fill rest with Udemy
        diverse_results = pd.concat([
            coursera_results.head(max(2, top_n // 4)),  # Get top Coursera courses (at least 2)
            udemy_results  # Get top Udemy courses
        ])
        # Sort combined results by similarity score and limit to top_n
        diverse_results = diverse_results.sort_values('similarity_score', ascending=False).head(top_n)
    else:
        # If one platform lacks relevant results, just return most similar courses
        # regardless of platform to ensure quality recommendations
        diverse_results = result.sort_values('similarity_score', ascending=False).head(top_n)
    
    # Return selected columns with course information and similarity score
    return diverse_results[[
        "course_title",  # Name of the course
        "course_organization",  # Platform or university offering the course
        "course_Certificate_type",  # Subject area or certificate type
        "course_rating",  # Average user rating
        "course_difficulty",  # Difficulty level
        "course_students_enrolled",  # Number of enrolled students
        "source",  # Platform (Coursera or Udemy)
        "similarity_score"  # Match percentage (0-100)
    ]]

@app.route('/')
def index():
    """
    Render the main application page.
    
    Brief:
        Flask route handler for the home page. Displays the search interface
        where users can enter topics and filter by platform.
    
    Usage:
        Accessed via: GET http://127.0.0.1:5000/
        
    Returns:
        str: Rendered HTML from 'templates/index.html'
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle course recommendation requests via POST.
    
    Brief:
        API endpoint that processes search queries and returns personalized
        course recommendations with optional platform filtering.
    
    Usage:
        POST http://127.0.0.1:5000/recommend
        Content-Type: application/json
        
        Request Body:
            {
                "query": "Python programming",
                "top_n": 10,
                "platform": "all"
            }
            
        Response:
            {
                "recommendations": [
                    {
                        "title": "...",
                        "organization": "...",
                        "similarity": 85.3,
                        ...
                    }
                ]
            }
    
    Returns:
        tuple: (JSON response, HTTP status code)
    """
    try:
        # Parse JSON request body
        data = request.get_json()
        
        # Extract search query and remove leading/trailing whitespace
        query = data.get('query', '').strip()
        
        # Extract number of results requested, default to 10
        top_n = int(data.get('top_n', 10))
        
        # Extract platform filter (all, Coursera, or Udemy)
        platform_filter = data.get('platform', 'all')
        
        # Validate that query is not empty
        if not query:
            return jsonify({'error': 'Please enter a topic or interest'}), 400
        
        # Get recommendations (request 3x to allow for platform filtering)
        recommendations = find_similar_courses(query, top_n * 3)
        
        # Apply platform filter if user selected specific platform
        if platform_filter != 'all':
            # Filter by selected platform and limit to requested count
            recommendations = recommendations[recommendations['source'] == platform_filter].head(top_n)
        else:
            # Return top N results from all platforms
            recommendations = recommendations.head(top_n)
        
        # Convert DataFrame to list of dictionaries for JSON response
        results = []
        recommended_indices = []  # Store indices for feedback tracking
        for idx, row in recommendations.iterrows():
            results.append({
                'title': row['course_title'],  # Course name
                'organization': row['course_organization'],  # Platform/university
                'category': row['course_Certificate_type'],  # Subject area
                'rating': float(row['course_rating']) if not pd.isna(row['course_rating']) else 0.0,  # Rating (handle NaN)
                'difficulty': row['course_difficulty'],  # Difficulty level
                'students': int(row['course_students_enrolled']) if not pd.isna(row['course_students_enrolled']) else 0,  # Enrollment (handle NaN)
                'source': row['source'],  # Platform identifier
                'similarity': float(row['similarity_score']),  # Match percentage
                'index': int(idx)  # Course index for feedback tracking
            })
            recommended_indices.append(int(idx))
        
        # Return recommendations with metadata for feedback
        return jsonify({
            'recommendations': results,
            'query': query,
            'recommended_indices': recommended_indices
        })
    
    except Exception as e:
        # Handle any errors and return error message
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """
    Get dataset statistics for dashboard display.
    
    Brief:
        API endpoint that returns aggregate statistics about the course database
        including course counts, platform distribution, and average ratings.
    
    Usage:
        GET http://127.0.0.1:5000/stats
        
        Response:
            {
                "total_courses": 4569,
                "coursera_courses": 891,
                "udemy_courses": 3678,
                "avg_rating": 4.2,
                "total_students": 15000000
            }
    
    Returns:
        flask.Response: JSON object containing database statistics
    """
    # Declare global df variable to access course data
    global df
    
    # Calculate and compile statistics
    stats_data = {
        'total_courses': len(df),  # Total number of courses in database
        'coursera_courses': len(df[df['source'] == 'Coursera']),  # Count of Coursera courses
        'udemy_courses': len(df[df['source'] == 'Udemy']),  # Count of Udemy courses
        'avg_rating': float(df['course_rating'].mean()),  # Average rating across all courses
        'total_students': int(df['course_students_enrolled'].sum())  # Total enrollment across all courses
    }
    
    # Return statistics as JSON
    return jsonify(stats_data)

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """
    Collect user feedback on course recommendations.
    
    Brief:
        API endpoint for collecting user feedback (thumbs up/down) on
        recommended courses. Stores feedback for evaluation and model improvement.
    
    Usage:
        POST http://127.0.0.1:5000/feedback
        Content-Type: application/json
        
        Request Body:
            {
                "query": "Python programming",
                "course_index": 123,
                "course_title": "Complete Python Bootcamp",
                "helpful": true,
                "recommended_indices": [123, 456, 789]
            }
        
        Response:
            {
                "status": "success",
                "message": "Feedback recorded"
            }
    
    Returns:
        tuple: (JSON response, HTTP status code)
    """
    global feedback_data
    
    try:
        # Parse JSON request body
        data = request.get_json()
        
        # Extract feedback information
        query = data.get('query', '').strip()
        course_index = data.get('course_index', -1)
        course_title = data.get('course_title', '')
        helpful = data.get('helpful', False)
        recommended_indices = data.get('recommended_indices', [])
        
        # Validate required fields
        if not query or course_index < 0:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create feedback entry
        feedback_entry = {
            'query': query,
            'course_index': course_index,
            'course_title': course_title,
            'helpful': helpful,
            'recommended_indices': recommended_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store feedback in memory
        feedback_data.append(feedback_entry)
        
        # Save feedback to file for persistence
        try:
            from evaluation import save_feedback
            save_feedback(feedback_data, '/app/data/feedback_data.json')
        except Exception as e:
            print(f"Warning: Could not save feedback to file: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded',
            'total_feedback': len(feedback_data)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    """
    Get evaluation metrics based on user feedback.
    
    Brief:
        API endpoint that calculates and returns model performance metrics
        including precision, recall, NDCG, MRR, and user satisfaction based
        on collected feedback data.
    
    Usage:
        GET http://127.0.0.1:5000/metrics?k=10
        
        Response:
            {
                "precision_at_k": 0.65,
                "recall_at_k": 0.82,
                "ndcg_at_k": 0.78,
                "mrr": 0.71,
                "user_satisfaction": 75.5,
                "total_queries": 25,
                "total_feedback": 150
            }
    
    Returns:
        flask.Response: JSON object containing evaluation metrics
    """
    global feedback_data
    
    try:
        # Get k parameter from query string (default: 10)
        k = int(request.args.get('k', 10))
        
        # Calculate metrics using evaluation module
        from evaluation import calculate_metrics_from_feedback
        metrics = calculate_metrics_from_feedback(feedback_data, k=k)
        
        return jsonify(metrics), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Application entry point
if __name__ == '__main__':
    # Print status message
    print("Loading course data...")
    
    # Load and prepare course data for recommendations
    load_and_prepare_data()
    
    # Load existing feedback data if available
    try:
        from evaluation import load_feedback
        feedback_data = load_feedback('/app/data/feedback_data.json')
        print(f"Loaded {len(feedback_data)} existing feedback entries")
    except Exception as e:
        print(f"No existing feedback data found: {e}")
        feedback_data = []
    
    # Print status message
    print("Starting Flask server...")
    
    # Determine if running in Docker
    import os
    debug_mode = os.getenv('FLASK_ENV', 'development') != 'production'
    
    # Start Flask server
    app.run(
        debug=debug_mode,  # Enable debug mode for development only
        host='0.0.0.0',  # Listen on all network interfaces (required for Docker)
        port=5000  # Use port 5000
    )
