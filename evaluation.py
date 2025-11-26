"""
Model Evaluation Module for Course Recommendation System

Author: Rana M Almuaied
Email: raanosh12@gmail.com
Date: November 26, 2025

Brief:
    Implements evaluation metrics as specified in the proposal to measure
    recommendation quality, including precision, recall, NDCG, and MRR.

Usage:
    >>> from evaluation import evaluate_recommendations, calculate_metrics
    >>> metrics = calculate_metrics(feedback_data)
    >>> print(f"Precision: {metrics['precision']:.2f}")
"""

# Import required libraries
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
from sklearn.metrics import precision_score, recall_score  # Evaluation metrics
from collections import defaultdict  # Data structure for grouping
import json  # JSON file operations


def calculate_precision_at_k(relevant_items, recommended_items, k=10):
    """
    Calculate Precision@K metric.
    
    Brief:
        Measures the proportion of relevant items in the top-K recommendations.
        Higher values indicate more accurate recommendations.
    
    Usage:
        >>> calculate_precision_at_k([1, 3, 5], [1, 2, 3, 4, 5], k=5)
        0.6
    
    Args:
        relevant_items (list): List of indices of relevant courses
        recommended_items (list): List of indices of recommended courses
        k (int): Number of top recommendations to consider
    
    Returns:
        float: Precision score (0.0 to 1.0)
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    # Take only top K recommendations
    recommended_k = recommended_items[:k]
    
    # Count how many recommended items are relevant
    relevant_count = len(set(recommended_k) & set(relevant_items))
    
    # Precision = (relevant items in top K) / K
    return relevant_count / k


def calculate_recall_at_k(relevant_items, recommended_items, k=10):
    """
    Calculate Recall@K metric.
    
    Brief:
        Measures the proportion of relevant items that were successfully
        retrieved in the top-K recommendations.
    
    Usage:
        >>> calculate_recall_at_k([1, 3, 5, 7], [1, 2, 3, 4, 5], k=5)
        0.75
    
    Args:
        relevant_items (list): List of indices of relevant courses
        recommended_items (list): List of indices of recommended courses
        k (int): Number of top recommendations to consider
    
    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # Take only top K recommendations
    recommended_k = recommended_items[:k]
    
    # Count how many relevant items were retrieved
    relevant_count = len(set(recommended_k) & set(relevant_items))
    
    # Recall = (relevant items retrieved) / (total relevant items)
    return relevant_count / len(relevant_items)


def calculate_ndcg_at_k(relevant_items, recommended_items, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Brief:
        Measures ranking quality considering position of relevant items.
        Items at higher positions contribute more to the score.
    
    Usage:
        >>> calculate_ndcg_at_k([1, 3, 5], [1, 2, 3, 4, 5], k=5)
        0.87
    
    Args:
        relevant_items (list): List of indices of relevant courses
        recommended_items (list): List of indices of recommended courses
        k (int): Number of top recommendations to consider
    
    Returns:
        float: NDCG score (0.0 to 1.0)
    """
    if k == 0 or len(relevant_items) == 0:
        return 0.0
    
    # Take only top K recommendations
    recommended_k = recommended_items[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            # Relevance = 1 if relevant, 0 otherwise
            # Discount by log2(position + 2) to emphasize higher ranks
            dcg += 1.0 / np.log2(i + 2)
    
    # Calculate ideal DCG (all relevant items at top)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    # Normalize DCG by ideal DCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_mrr(relevant_items, recommended_items):
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Brief:
        Measures how early the first relevant item appears in recommendations.
        Higher values indicate relevant items are ranked higher.
    
    Usage:
        >>> calculate_mrr([3, 5], [1, 2, 3, 4, 5])
        0.333  # First relevant item is at position 3
    
    Args:
        relevant_items (list): List of indices of relevant courses
        recommended_items (list): List of indices of recommended courses
    
    Returns:
        float: MRR score (0.0 to 1.0)
    """
    # Find position of first relevant item
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            # Reciprocal rank = 1 / (position + 1)
            return 1.0 / (i + 1)
    
    # No relevant items found
    return 0.0


def calculate_metrics_from_feedback(feedback_data, k=10):
    """
    Calculate aggregate evaluation metrics from user feedback data.
    
    Brief:
        Processes user feedback to compute precision, recall, NDCG, MRR,
        and user satisfaction scores for model evaluation.
    
    Usage:
        >>> feedback = load_feedback_data()
        >>> metrics = calculate_metrics_from_feedback(feedback, k=10)
        >>> print(metrics)
    
    Args:
        feedback_data (list): List of feedback dictionaries with keys:
            - query: search query
            - recommended_courses: list of recommended course indices
            - relevant_courses: list of indices marked as helpful
        k (int): Number of top recommendations to evaluate
    
    Returns:
        dict: Dictionary containing:
            - precision_at_k: Average precision@K
            - recall_at_k: Average recall@K
            - ndcg_at_k: Average NDCG@K
            - mrr: Mean Reciprocal Rank
            - user_satisfaction: Percentage of positive feedback
            - total_queries: Number of evaluated queries
    """
    if not feedback_data:
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'mrr': 0.0,
            'user_satisfaction': 0.0,
            'total_queries': 0
        }
    
    # Initialize metric accumulators
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    mrr_scores = []
    positive_feedback_count = 0
    
    # Group feedback by query
    query_feedback = defaultdict(lambda: {'recommended': [], 'relevant': []})
    
    for feedback in feedback_data:
        query = feedback.get('query', '')
        course_idx = feedback.get('course_index', -1)
        is_helpful = feedback.get('helpful', False)
        recommended = feedback.get('recommended_indices', [])
        
        if query and course_idx >= 0:
            query_feedback[query]['recommended'] = recommended
            if is_helpful:
                query_feedback[query]['relevant'].append(course_idx)
                positive_feedback_count += 1
    
    # Calculate metrics for each query
    for query, data in query_feedback.items():
        recommended = data['recommended']
        relevant = data['relevant']
        
        if len(recommended) > 0 and len(relevant) > 0:
            precision_scores.append(calculate_precision_at_k(relevant, recommended, k))
            recall_scores.append(calculate_recall_at_k(relevant, recommended, k))
            ndcg_scores.append(calculate_ndcg_at_k(relevant, recommended, k))
            mrr_scores.append(calculate_mrr(relevant, recommended))
    
    # Calculate average metrics
    total_feedback = len(feedback_data)
    
    return {
        'precision_at_k': np.mean(precision_scores) if precision_scores else 0.0,
        'recall_at_k': np.mean(recall_scores) if recall_scores else 0.0,
        'ndcg_at_k': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'user_satisfaction': (positive_feedback_count / total_feedback * 100) if total_feedback > 0 else 0.0,
        'total_queries': len(query_feedback),
        'total_feedback': total_feedback
    }


def save_feedback(feedback_data, filepath='feedback_data.json'):
    """
    Save feedback data to JSON file.
    
    Brief:
        Persists user feedback to disk for later analysis and evaluation.
    
    Args:
        feedback_data (list): List of feedback dictionaries
        filepath (str): Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(feedback_data, f, indent=2)


def load_feedback(filepath='feedback_data.json'):
    """
    Load feedback data from JSON file.
    
    Brief:
        Retrieves previously saved user feedback for evaluation.
    
    Args:
        filepath (str): Path to JSON file
    
    Returns:
        list: List of feedback dictionaries
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    # Example usage and testing
    print("Course Recommendation System - Evaluation Module")
    print("=" * 50)
    
    # Example data
    example_feedback = [
        {
            'query': 'Python programming',
            'course_index': 1,
            'helpful': True,
            'recommended_indices': [1, 2, 3, 4, 5]
        },
        {
            'query': 'Python programming',
            'course_index': 3,
            'helpful': True,
            'recommended_indices': [1, 2, 3, 4, 5]
        },
        {
            'query': 'Machine Learning',
            'course_index': 10,
            'helpful': True,
            'recommended_indices': [8, 9, 10, 11, 12]
        }
    ]
    
    metrics = calculate_metrics_from_feedback(example_feedback, k=5)
    
    print("\nEvaluation Metrics:")
    print(f"Precision@5: {metrics['precision_at_k']:.3f}")
    print(f"Recall@5: {metrics['recall_at_k']:.3f}")
    print(f"NDCG@5: {metrics['ndcg_at_k']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"User Satisfaction: {metrics['user_satisfaction']:.1f}%")
    print(f"Total Queries Evaluated: {metrics['total_queries']}")
