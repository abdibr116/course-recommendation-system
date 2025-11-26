"""
Course Recommendation System - Testing Script

Author: Rana M Almuaied
Email: raanosh12@gmail.com
Date: November 26, 2025

Brief:
    Standalone testing script for course recommendation algorithm.
    Implements TF-IDF vectorization and K-NN similarity matching
    for finding similar courses based on user queries.

Usage:
    Run directly to test recommendation system interactively:
    >>> python test/load_dataset.py
    
    Or import and use functions:
    >>> from test.load_dataset import find_similar_courses_by_title
    >>> results = find_similar_courses_by_title("Python programming", top_n=5)
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack


# ==========================================================
# 1. LOAD DATA
# ==========================================================

# Load Coursera data
df_coursera = pd.read_csv("../dataset/coursea_data.csv")

# Load Udemy data
df_udemy = pd.read_csv("../dataset/udemy_courses.csv")

# Normalize Udemy columns to match Coursera structure
df_udemy_normalized = pd.DataFrame({
    'course_title': df_udemy['course_title'],
    'course_organization': 'Udemy',  # All Udemy courses
    'course_Certificate_type': df_udemy['subject'],  # Use subject as certificate type
    'course_rating': pd.to_numeric(df_udemy.get('course_rating', 0), errors='coerce'),  # If available
    'course_difficulty': df_udemy['level'],
    'course_students_enrolled': df_udemy['num_subscribers'],
    'source': 'Udemy'
})

# Add source column to Coursera data
df_coursera['source'] = 'Coursera'

# Combine both datasets
df = pd.concat([df_coursera, df_udemy_normalized], ignore_index=True)

print(f"Dataset loaded. Total courses: {len(df)}")
print(f"  - Coursera: {len(df_coursera)}")
print(f"  - Udemy: {len(df_udemy_normalized)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())


# ==========================================================
# 2. CLEAN & PREPROCESS NUMERIC FIELDS
# ==========================================================

# ---------- helper: parse '5.3k', '1.2M', '800', etc. ----------
def parse_number(x):
    """
    Convert values like:
        '5.3k' -> 5300
        '1.2M' -> 1200000
        '800'  -> 800
        '12,345' -> 12345
    into floats.
    """
    if pd.isna(x):
        return np.nan

    if not isinstance(x, str):
        # already numeric
        return float(x)

    s = x.strip().lower()

    # remove words like "students", "enrolled", etc. if they exist
    for word in ["students", "enrolled", "student", " ", "+"]:
        s = s.replace(word, "")

    # remove commas
    s = s.replace(",", "")

    # handle k / m suffix
    if s.endswith("k"):
        num = float(s[:-1]) * 1000
    elif s.endswith("m"):
        num = float(s[:-1]) * 1_000_000
    else:
        num = float(s)

    return num


# ---------- clean course_students_enrolled ----------
df["course_students_enrolled"] = df["course_students_enrolled"].apply(parse_number)

# ---------- clean course_rating ----------
# if rating is like "4.8" as string, convert to float
df["course_rating"] = pd.to_numeric(df["course_rating"], errors="coerce")

# numeric columns we will use
numeric_cols = ["course_rating", "course_students_enrolled"]

# fill missing numeric values with 0
df[numeric_cols] = df[numeric_cols].fillna(0.0)

# scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numeric_cols])


# ==========================================================
# 3. BUILD TEXT FEATURES
# ==========================================================

# Create one combined text field for similarity
df["text"] = (
    df["course_title"].fillna("") + " " +
    df["course_organization"].fillna("") + " " +
    df["course_Certificate_type"].fillna("") + " " +
    df["course_difficulty"].fillna("")
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_text = vectorizer.fit_transform(df["text"])


# ==========================================================
# 4. COMBINE TEXT + NUMERIC FEATURES
# ==========================================================

X_combined = hstack([X_text, X_num])
X_combined = X_combined.tocsr()   # <-- FIX HERE

# ==========================================================
# 5. BUILD SIMILARITY MODEL (COSINE)
# ==========================================================

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(X_combined)


# ==========================================================
# 6. HELPER FUNCTIONS
# ==========================================================

def find_similar_courses_by_index(course_index, top_n=5):
    """
    Find courses similar to the course at the given row index.
    """
    query_vec = X_combined[course_index]
    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n + 1)

    # first neighbor is the course itself → skip it
    similar_idx = indices[0][1:]
    similar_dist = distances[0][1:]

    result = df.iloc[similar_idx].copy()
    result["distance"] = similar_dist
    result["similarity_score"] = 1 - similar_dist  # cosine: 1 - distance

    return result[[
        "course_title",
        "course_organization",
        "course_Certificate_type",
        "course_rating",
        "course_difficulty",
        "course_students_enrolled",
        "source",
        "similarity_score"
    ]]


def find_similar_courses_by_title(title_query, top_n=5):
    """
    Find similar courses given a text query (e.g. 'Python for Data Science').
    The query does not need to exist in the dataset.
    """
    # build text vector for query
    query_text = str(title_query)
    query_vec_text = vectorizer.transform([query_text])

    # for numeric features in query we don't really have info,
    # so we use zeros (mean after scaling)
    zero_num = np.zeros((1, X_num.shape[1]))

    query_vec = hstack([query_vec_text, zero_num])

    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n)

    result = df.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    result["similarity_score"] = 1 - distances[0]

    return result[[
        "course_title",
        "course_organization",
        "course_Certificate_type",
        "course_rating",
        "course_difficulty",
        "course_students_enrolled",
        "source",
        "similarity_score"
    ]]


# ==========================================================
# 7. QUICK TEST
# ==========================================================

if __name__ == "__main__":
    # test 1: by index
    test_index = 15  # change this to any index within df
    print("\n=============================")
    print("COURSE SELECTED (by index):")
    print(df.iloc[test_index][["course_title", "course_organization", "course_rating", "source"]])
    print("=============================\n")

    similar = find_similar_courses_by_index(test_index, top_n=5)
    print("Similar courses (by index):")
    print(similar)

    # test 2: by text query
    query = "Python for data science"
    print("\n=============================")
    print(f"COURSE QUERY (by text): {query}")
    print("=============================\n")

    similar_q = find_similar_courses_by_title(query, top_n=5)
    print("Similar courses (by text query):")
    print(similar_q)

    # Test 3: interactive text query from user
    user_query = input(
        "\nEnter a course title or description to find similar courses "
        "(or press Enter to skip): "
    ).strip()

    if user_query:
        similar_q = find_similar_courses_by_title(user_query, top_n=5)
        print("\nSimilar courses (by your query):")
        print(similar_q)
    else:
        print("\nNo query entered, ending.")
