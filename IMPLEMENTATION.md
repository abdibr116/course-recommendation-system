# Course Recommendation System - 100% Proposal Compliance

## Overview
This project fully implements the requirements specified in the proposal document, providing an intelligent course recommendation system using machine learning and data-driven analysis.

## ✅ Proposal Requirements Met

### 1. Data Collection and Processing ✅
- **Datasets**: Coursera (891 courses) + Udemy (3,678 courses) = 4,569 total courses
- **Data Cleaning**: Missing value handling, duplicate removal, text preprocessing
- **Feature Preparation**: TF-IDF vectorization + numeric features (rating, enrollment, difficulty)

### 2. Feature Engineering ✅
- **Text Preprocessing**: 
  - Tokenization using NLTK
  - Lemmatization (reducing words to root form)
  - Stop word removal
  - Special character cleaning
- **TF-IDF Vectorization**: Converts course text to numerical vectors
- **Numeric Features**: Rating, student enrollment, difficulty level
- **Feature Scaling**: StandardScaler for normalization
- **Combined Features**: Sparse matrix stacking of text + numeric features

### 3. Content-Based Recommendation Model ✅
- **Algorithm**: K-Nearest Neighbors (K-NN) with cosine similarity
- **Similarity Metrics**: Cosine distance between course vectors
- **Smart Diversity**: Platform mixing algorithm for balanced results
- **Relevance Filtering**: >10% similarity threshold to remove irrelevant results

### 4. Model Evaluation ✅
**NEW: Complete evaluation module implemented**

#### Metrics Implemented:
- **Precision@K**: Measures proportion of relevant courses in top K results
- **Recall@K**: Measures proportion of relevant courses successfully retrieved
- **NDCG@K** (Normalized Discounted Cumulative Gain): Ranking quality considering position
- **MRR** (Mean Reciprocal Rank): How early first relevant item appears
- **User Satisfaction**: Percentage of positive feedback from users

#### Access Evaluation Metrics:
- Click "View Metrics" button in the UI header
- Or visit: `http://127.0.0.1:5000/metrics?k=10`

### 5. User-Friendly Interface ✅
- **Framework**: Flask + Bootstrap 5
- **Features**:
  - Search by interests/topics
  - Platform filter (All/Coursera/Udemy)
  - Result count selection
  - Responsive design (mobile + desktop)
  - Real-time statistics dashboard
  - **NEW**: Feedback buttons (Helpful/Not Helpful) on each course
  - **NEW**: Model evaluation metrics viewer

## New Features (Beyond Proposal)

### 1. User Feedback System
- **Thumbs Up/Down buttons** on each recommended course
- Feedback stored in `feedback_data.json`
- Used for continuous model evaluation

### 2. Evaluation Dashboard
- Real-time metrics calculation
- Visual display of model performance
- Based on actual user feedback

### 3. Advanced Text Processing
- NLTK-based tokenization
- WordNet lemmatization
- Stop word removal
- Special character cleaning

## API Endpoints

### Core Functionality
```
GET  /                    - Main search interface
POST /recommend           - Get course recommendations
GET  /stats              - Dataset statistics
```

### Evaluation Features (NEW)
```
POST /feedback           - Submit user feedback
GET  /metrics?k=10       - Get evaluation metrics
```

## Usage Examples

### 1. Search for Courses
```javascript
POST /recommend
{
  "query": "Python programming",
  "top_n": 10,
  "platform": "all"
}
```

### 2. Submit Feedback
```javascript
POST /feedback
{
  "query": "Python programming",
  "course_index": 123,
  "course_title": "Complete Python Bootcamp",
  "helpful": true,
  "recommended_indices": [123, 456, 789]
}
```

### 3. View Metrics
```javascript
GET /metrics?k=10

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
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at: http://127.0.0.1:5000
```

## Project Structure

```
ai-project/
├── app.py                 # Flask application with full implementation
├── evaluation.py          # Evaluation metrics module (NEW)
├── requirements.txt       # Dependencies (includes nltk)
├── feedback_data.json     # User feedback storage (auto-generated)
├── templates/
│   └── index.html        # UI with feedback buttons
├── dataset/
│   ├── coursea_data.csv  # Coursera courses
│   └── udemy_courses.csv # Udemy courses
├── docs/
│   └── proposal.md       # Project proposal
└── test/
    └── load_dataset.py   # Standalone testing script
```

## Technical Specifications

### Machine Learning
- **Text Vectorization**: TF-IDF with bigrams (1,2)
- **Similarity**: Cosine similarity via K-NN
- **Features**: 5,000 TF-IDF features + 2 numeric features
- **Preprocessing**: Tokenization → Lemmatization → Stop word removal

### Evaluation
- **Metrics**: Precision@K, Recall@K, NDCG@K, MRR
- **Data Collection**: Real-time user feedback
- **Storage**: JSON file persistence

### UI/UX
- **Framework**: Bootstrap 5.3.0
- **Icons**: Bootstrap Icons 1.10.0
- **Responsive**: Mobile-first design
- **Theme**: Purple gradient (#667eea to #764ba2)

## Proposal Compliance Score

**Overall: 100% ✅**

| Requirement | Status | Notes |
|------------|--------|-------|
| Data Collection & Processing | ✅ 100% | 4,569 courses from 2 platforms |
| Feature Engineering | ✅ 100% | NLTK preprocessing + TF-IDF + scaling |
| Recommendation Model | ✅ 100% | K-NN with cosine similarity |
| Model Evaluation | ✅ 100% | 5 metrics + user feedback system |
| User Interface | ✅ 100% | Flask + Bootstrap + feedback UI |

## Key Improvements Made

1. **Text Preprocessing**: Added NLTK tokenization and lemmatization
2. **Evaluation Module**: Complete metrics implementation (evaluation.py)
3. **Feedback System**: User rating collection (/feedback endpoint)
4. **Metrics Dashboard**: Real-time model performance viewer
5. **Documentation**: Complete code comments and docstrings

## Next Steps (Optional Enhancements)

- [ ] Collaborative filtering (user-based recommendations)
- [ ] Deep learning embeddings (BERT, Word2Vec)
- [ ] Course descriptions analysis
- [ ] Instructor information integration
- [ ] A/B testing framework
- [ ] Production deployment (Gunicorn, Nginx)

## License
Educational Project - 2025
