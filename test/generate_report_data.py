"""
Generate Report Data and Statistics for Final Project Report

Author: Rana M Almuaied
Email: raanosh12@gmail.com
Date: November 26, 2025

Brief:
    This script generates comprehensive statistics and visualizations
    for the final project report, including dataset analysis, model
    performance metrics, and evaluation results.

Usage:
    python test/generate_report_data.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import (
    calculate_metrics_from_feedback,
    load_feedback
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create generated directory if it doesn't exist
os.makedirs('generated', exist_ok=True)


def analyze_dataset():
    """
    Analyze the combined dataset and generate statistics.
    
    Returns:
        dict: Dataset statistics
    """
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    df_coursera = pd.read_csv('dataset/coursea_data.csv')
    df_udemy = pd.read_csv('dataset/udemy_courses.csv')
    
    stats = {
        'total_courses': len(df_coursera) + len(df_udemy),
        'coursera_courses': len(df_coursera),
        'udemy_courses': len(df_udemy),
        'coursera_avg_rating': df_coursera['Rating'].mean(),
        'udemy_avg_subscribers': df_udemy['num_subscribers'].mean(),
    }
    
    print(f"\nTotal Courses: {stats['total_courses']:,}")
    print(f"  - Coursera: {stats['coursera_courses']:,}")
    print(f"  - Udemy: {stats['udemy_courses']:,}")
    print(f"\nAverage Rating (Coursera): {stats['coursera_avg_rating']:.2f}/5.0")
    print(f"Average Subscribers (Udemy): {stats['udemy_avg_subscribers']:,.0f}")
    
    # Analyze course difficulty distribution
    print("\nCourse Difficulty Distribution (Coursera):")
    difficulty_dist = df_coursera['Level'].value_counts()
    for level, count in difficulty_dist.items():
        percentage = (count / len(df_coursera)) * 100
        print(f"  - {level}: {count} ({percentage:.1f}%)")
    
    # Analyze top organizations
    print("\nTop 10 Course Providers (Coursera):")
    top_orgs = df_coursera['Offered By'].value_counts().head(10)
    for org, count in top_orgs.items():
        print(f"  - {org}: {count} courses")
    
    return stats


def analyze_text_preprocessing():
    """
    Demonstrate text preprocessing pipeline.
    
    Returns:
        dict: Preprocessing examples
    """
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING ANALYSIS")
    print("=" * 60)
    
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Example course title
    example_text = "Machine Learning: Classification and Regression Analysis"
    
    print(f"\nOriginal: {example_text}")
    
    # Step by step preprocessing
    text = example_text.lower()
    print(f"Lowercase: {text}")
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    print(f"Remove special chars: {text}")
    
    tokens = text.split()
    print(f"Tokenization: {tokens}")
    
    tokens = [word for word in tokens if word not in stop_words]
    print(f"Stop word removal: {tokens}")
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    print(f"Lemmatization: {tokens}")
    
    return {
        'original': example_text,
        'preprocessed': ' '.join(tokens)
    }


def analyze_model_performance():
    """
    Analyze recommendation model performance using feedback data.
    
    Returns:
        dict: Performance metrics
    """
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    try:
        # Load feedback data
        feedback_data = load_feedback('/app/data/feedback_data.json')
        
        if not feedback_data:
            print("\nNo feedback data available yet.")
            print("Using simulated example data for demonstration...")
            
            # Simulated feedback for demonstration
            feedback_data = [
                {
                    'query': 'Python programming',
                    'course_index': 1,
                    'helpful': True,
                    'recommended_indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                },
                {
                    'query': 'Python programming',
                    'course_index': 2,
                    'helpful': True,
                    'recommended_indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                },
                {
                    'query': 'Python programming',
                    'course_index': 5,
                    'helpful': True,
                    'recommended_indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                },
                {
                    'query': 'Machine Learning',
                    'course_index': 12,
                    'helpful': True,
                    'recommended_indices': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                },
                {
                    'query': 'Web Development',
                    'course_index': 20,
                    'helpful': True,
                    'recommended_indices': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                }
            ]
        
        print(f"\nTotal feedback entries: {len(feedback_data)}")
        
        # Calculate metrics for different K values
        k_values = [5, 10, 20]
        results = {}
        
        for k in k_values:
            metrics = calculate_metrics_from_feedback(feedback_data, k=k)
            results[k] = metrics
            
            print(f"\nMetrics @ K={k}:")
            print(f"  - Precision@{k}: {metrics['precision_at_k']:.3f}")
            print(f"  - Recall@{k}: {metrics['recall_at_k']:.3f}")
            print(f"  - NDCG@{k}: {metrics['ndcg_at_k']:.3f}")
            print(f"  - MRR: {metrics['mrr']:.3f}")
            print(f"  - User Satisfaction: {metrics['user_satisfaction']:.1f}%")
            print(f"  - Unique Queries: {metrics['total_queries']}")
        
        return results
        
    except Exception as e:
        print(f"\nError loading feedback: {e}")
        return {}


def generate_visualizations():
    """
    Generate visualizations for the report.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load datasets
    df_coursera = pd.read_csv('dataset/coursea_data.csv')
    df_udemy = pd.read_csv('dataset/udemy_courses.csv')
    
    # 1. Platform distribution
    plt.figure(figsize=(8, 6))
    platforms = ['Coursera', 'Udemy']
    counts = [len(df_coursera), len(df_udemy)]
    colors = ['#0056D2', '#EC5252']
    plt.bar(platforms, counts, color=colors)
    plt.title('Course Distribution by Platform', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Courses', fontsize=12)
    plt.xlabel('Platform', fontsize=12)
    for i, v in enumerate(counts):
        plt.text(i, v + 100, f'{v:,}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig('generated/platform_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: generated/platform_distribution.png")
    plt.close()
    
    # 2. Difficulty distribution (Coursera)
    plt.figure(figsize=(10, 6))
    difficulty_counts = df_coursera['Level'].value_counts()
    colors_diff = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    plt.pie(difficulty_counts.values, labels=difficulty_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=colors_diff)
    plt.title('Course Difficulty Distribution (Coursera)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('generated/difficulty_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: generated/difficulty_distribution.png")
    plt.close()
    
    # 3. Rating distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coursera ratings
    ax1.hist(df_coursera['Rating'].dropna(), bins=20, color='#0056D2', alpha=0.7, edgecolor='black')
    ax1.set_title('Coursera Rating Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rating', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.axvline(df_coursera['Rating'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_coursera["Rating"].mean():.2f}')
    ax1.legend()
    
    # Udemy subscribers
    ax2.hist(df_udemy['num_subscribers'].dropna(), bins=20, color='#EC5252', alpha=0.7, edgecolor='black')
    ax2.set_title('Udemy Subscribers Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Subscribers', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.axvline(df_udemy['num_subscribers'].mean(), color='red', linestyle='--',
                label=f'Mean: {df_udemy["num_subscribers"].mean():,.0f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('generated/rating_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: generated/rating_distribution.png")
    plt.close()
    
    print("\nAll visualizations saved to generated/ directory")


def generate_summary_report():
    """
    Generate a summary report in JSON format.
    """
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    dataset_stats = analyze_dataset()
    preprocessing_demo = analyze_text_preprocessing()
    model_performance = analyze_model_performance()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'dataset_statistics': dataset_stats,
        'preprocessing_example': preprocessing_demo,
        'model_performance': model_performance,
        'implementation_details': {
            'algorithm': 'Content-Based Filtering with TF-IDF',
            'similarity_metric': 'Cosine Similarity',
            'features': ['course_title', 'course_organization', 'course_category', 'course_difficulty'],
            'preprocessing_steps': ['Lowercase conversion', 'Special character removal', 'Tokenization', 
                                   'Stopword removal', 'Lemmatization'],
            'vectorization': 'TF-IDF (max_features=5000, ngram_range=(1,2))',
            'k_neighbors': 10
        }
    }
    
    # Save report
    with open('generated/report_data.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Summary report saved to: generated/report_data.json")
    
    return report


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COURSE RECOMMENDATION SYSTEM - REPORT DATA GENERATOR")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate all analyses
    generate_summary_report()
    generate_visualizations()
    
    print("\n" + "=" * 60)
    print("REPORT DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - generated/report_data.json")
    print("  - generated/platform_distribution.png")
    print("  - generated/difficulty_distribution.png")
    print("  - generated/rating_distribution.png")
    print("\nThese files contain data and visualizations for your final report.")
