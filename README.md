# Course Recommendation System

A web-based course recommendation system that helps users discover relevant courses from Coursera and Udemy based on their interests and preferred topics.

## Features

- 🎯 **Smart Recommendations**: Uses TF-IDF and K-Nearest Neighbors for content-based filtering
- 📊 **Dual Platform Support**: Searches across both Coursera (891 courses) and Udemy (3,678 courses)
- 🎨 **Modern UI**: Beautiful Bootstrap-based interface with gradient design
- 📈 **Real-time Statistics**: Shows total courses, platform distribution, and average ratings
- 🔍 **Flexible Search**: Enter any topic or interest to get personalized recommendations
- ⭐ **Detailed Results**: Displays course ratings, difficulty, enrollment numbers, and match percentage

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure datasets are in place:**
   - `dataset/coursea_data.csv`
   - `dataset/udemy_courses.csv`

## Usage

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Enter your interests** in the search box (e.g., "Python programming", "Machine Learning", "Web Development")

4. **View recommendations** with similarity scores, ratings, and detailed course information

## Project Structure

```
ai-project/
├── app.py                      # Flask application
├── templates/
│   └── index.html             # Main UI template
├── libs/
│   └── load_dataset.py        # Data loading and processing module
├── dataset/
│   ├── coursea_data.csv       # Coursera courses dataset
│   └── udemy_courses.csv      # Udemy courses dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How It Works

1. **Data Loading**: Combines Coursera and Udemy datasets with normalized schema
2. **Feature Extraction**: Creates TF-IDF vectors from course titles, organizations, categories, and difficulty levels
3. **Similarity Matching**: Uses cosine similarity to find courses most relevant to user queries
4. **Ranking**: Returns top N courses with similarity scores displayed as match percentages

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (TF-IDF, K-Nearest Neighbors)
- **Data Processing**: pandas, numpy
- **Frontend**: Bootstrap 5, Bootstrap Icons
- **UI Design**: Custom CSS with gradient backgrounds and smooth animations

## API Endpoints

- `GET /` - Main application page
- `POST /recommend` - Get course recommendations (JSON)
- `GET /stats` - Get dataset statistics (JSON)

## Example Queries

- "Python programming"
- "Machine Learning"
- "Web Development"
- "Data Science"
- "Business Finance"
- "Digital Marketing"

## License

This project is for educational purposes.
