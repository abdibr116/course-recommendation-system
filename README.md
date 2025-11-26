# Course Recommendation System

A web-based course recommendation system that helps users discover relevant courses from Coursera and Udemy based on their interests and preferred topics.

## Features

- 🎯 **Smart Recommendations**: Uses TF-IDF and K-Nearest Neighbors for content-based filtering
- 📊 **Dual Platform Support**: Searches across Coursera (8,370 courses) and Udemy (3,678 courses)
- 🎨 **Modern UI**: Beautiful Bootstrap-based interface with gradient design
- 📈 **Real-time Statistics**: Shows total courses, platform distribution, and average ratings
- 🔍 **Flexible Search**: Enter any topic or interest to get personalized recommendations
- ⭐ **Detailed Results**: Displays course ratings, difficulty, enrollment numbers, and match percentage
- 🐳 **Docker Ready**: Fully containerized for easy deployment

## Quick Start

### Prerequisites
- Docker
- Docker Compose

### Run with Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdibr116/course-recommendation-system.git
   cd course-recommendation-system
   ```

2. **Start the application:**
   ```bash
   docker compose up -d
   ```

3. **Access the application:**
   - Web Interface: http://localhost:5000
   - API Stats: http://localhost:5000/stats

4. **View logs:**
   ```bash
   docker compose logs -f
   ```

5. **Stop the application:**
   ```bash
   docker compose down
   ```

### Configuration

Change the port by creating a `.env` file:
```bash
# .env
PORT=8080
```

Then restart:
```bash
docker compose up -d
```

## Docker Commands

### Build and Run
```bash
# Build and start in detached mode
docker compose up -d --build

# View running containers
docker compose ps

# Follow logs
docker compose logs -f course-recommender
```

### Development
```bash
# Rebuild after code changes
docker compose up --build -d

# Stop and remove containers
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Health Check
The container includes a health check that runs every 30 seconds:
```bash
# Check container health status
docker compose ps
```

## Project Structure

```
course-recommendation-system/
├── app.py                      # Main Flask application
├── evaluation.py               # Evaluation metrics module
├── compose.yaml                # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── templates/
│   └── index.html             # Web UI template
├── dataset/
│   ├── coursea_data.csv       # Coursera courses (8,370 courses)
│   └── udemy_courses.csv      # Udemy courses (3,678 courses)
└── test/
    └── load_dataset.py        # Dataset testing utilities
```

## How It Works

1. **Data Loading**: Combines Coursera and Udemy datasets (12,048 total courses) with normalized schema
2. **Text Preprocessing**: Tokenization, stopword removal, and lemmatization using NLTK
3. **Feature Extraction**: Creates TF-IDF vectors from course titles, organizations, categories, and difficulty levels
4. **Similarity Matching**: Uses cosine similarity to find courses most relevant to user queries
5. **Ranking**: Returns top N courses with similarity scores displayed as match percentages

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (TF-IDF, K-Nearest Neighbors)
- **NLP**: NLTK (Natural Language Toolkit)
- **Data Processing**: pandas, numpy
- **Frontend**: Bootstrap 5, Bootstrap Icons
- **Containerization**: Docker, Docker Compose
- **UI Design**: Custom CSS with gradient backgrounds and smooth animations

## API Endpoints

- `GET /` - Main application page
- `POST /recommend` - Get course recommendations (JSON)
  ```json
  {
    "user_input": "Python programming",
    "k": 10
  }
  ```
- `GET /stats` - Get dataset statistics (JSON)

## Example Queries

- "Python programming"
- "Machine Learning"
- "Web Development"
- "Data Science"
- "Business Finance"
- "Digital Marketing"
- "Cloud Computing"

## Production Deployment

For production environments, consider:

1. **Use a production WSGI server** (Gunicorn):
   ```dockerfile
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

2. **Add reverse proxy** (Nginx) for HTTPS and load balancing

3. **Set resource limits** in compose.yaml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

4. **Enable monitoring** with health checks and logging

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs course-recommender

# Inspect container
docker inspect course-recommendation-system
```

### Port already in use
Create a `.env` file:
```bash
echo "PORT=8080" > .env
docker compose up -d
```

### Out of memory
Increase Docker memory limit in Docker Desktop settings or add memory limits in compose.yaml.

## Author

- **Name**: Rana M Almuaied
- **Email**: raanosh12@gmail.com
- **Date**: November 26, 2025

## License

This project is for educational purposes.
