from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource
def _get_embedding_model():
    """
    Load the embedding model with Streamlit caching.
    
    Uses @st.cache_resource because this is a global resource (ML model)
    that should be loaded once and reused across sessions.
    """
    try:
        from sentence_transformers import SentenceTransformer
        # Use fast, lightweight model optimized for speed
        # BAAI/bge-small-en-v1.5 great model for single words or skills.
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        logger.info("Embedding model loaded successfully")
        return model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        raise
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

def compute_skill_gap(rows, user_skills, skill_levels=None):
    """
    Compute skill gap considering user skills and optionally their levels.
    
    Args:
        rows: List of job dictionaries with 'skills_detected' key
        user_skills: Set or list of skills the user has
        skill_levels: Dict mapping skill -> level (can be numeric 1-4 or string: "Basic", "Intermediate", "Advanced", "Expert")
    
    Returns:
        Tuple of (rows with match metrics, missing skills list)
    """
    user_skills = set(user_skills) if user_skills else set()
    skill_levels = skill_levels or {}
    
    # Map string level names to numeric values (for backward compatibility with numeric 1-4)
    level_name_to_numeric = {
        "Basic": 1,
        "Intermediate": 2,
        "Advanced": 3,
        "Expert": 4
    }
    
    def get_numeric_level(skill):
        """Convert skill level to numeric (1-4). Handles both string and numeric inputs."""
        level = skill_levels.get(skill, None)
        if level is None: #happens for custom skills
            return 4  # Default to Expert (4) if not specified
        # if it's already a number, return it
        if isinstance(level, (int, float)):
            return int(level)
        # If it's a string, convert using mapping
        if isinstance(level, str):
            return level_name_to_numeric.get(level, 4)  # Default to Expert if unknown string
        return 4  # Fallback to Expert

    for row in rows:
        job_sk = set(row["skills_detected"])
        row["n_skills_job"] = len(job_sk)
        
        # Count skills user has (binary)
        row["n_skills_user_has"] = len(job_sk & user_skills)
        
        # Calculate weighted match considering skill levels
        # Skills with higher levels contribute more to the match
        weighted_match = 0.0
        total_weight = 0.0
        
        for skill in job_sk:
            if skill in user_skills:
                # User has the skill - weight by level (1-4, normalized to 0.25-1.0)
                user_level_numeric = get_numeric_level(skill)
                weight = user_level_numeric / 4.0  # Normalize to 0.25-1.0
                weighted_match += weight
                total_weight += 1.0
            else:
                # User doesn't have the skill
                total_weight += 1.0
        
        # Match ratio: binary (original)
        row["match_ratio"] = (
            row["n_skills_user_has"] / row["n_skills_job"]
            if row["n_skills_job"] > 0 else 0
        )
        
        # Weighted match ratio: considers skill levels (only of users)
        row["weighted_match_ratio"] = (
            weighted_match / total_weight if total_weight > 0 else 0
        )
        
        # Average level of matched skills (numeric 1-4)
        matched_levels = [get_numeric_level(skill) 
                         for skill in job_sk if skill in user_skills]
        row["avg_skill_level"] = (
            sum(matched_levels) / len(matched_levels) 
            if matched_levels else 0
        )

    all_skills = []
    for row in rows:
        all_skills.extend(row["skills_detected"])

    freq = Counter(all_skills)

    # store missing skills with their counts and priority (here priority = count)
    missing = [
        {
            "skill": skill,
            "count": count,
            "priority": count
        }
        for skill, count in freq.most_common()
        if skill not in user_skills
    ]

    return rows, missing


def cluster_skills_dynamic(all_skills: list, n_clusters: int = None, max_clusters: int = 8) -> dict:
    """
    Dynamically cluster skills using embeddings and KMeans.
    
    This function creates embeddings for all unique skills found in the query results
    and clusters them into coherent groups based on semantic similarity.
    
    Args:
        all_skills: List of unique skills (strings)
        n_clusters: Optional number of clusters. If None, auto-determines based on data size
        max_clusters: Maximum number of clusters to create (default: 8)
        
    Returns:
        Dictionary mapping skill -> cluster_id
    """
    start_time = time.time()
    
    if not all_skills or len(all_skills) == 0:
        return {}
    
    # Remove duplicates while preserving order
    unique_skills = list(dict.fromkeys(all_skills))
    
    if len(unique_skills) == 1:
        return {unique_skills[0]: 0}
    
    try:
        model = _get_embedding_model()
        
        # Generate embeddings for all skills
        embeddings = model.encode(unique_skills, show_progress_bar=False, convert_to_numpy=True)
        
        # Auto-determine number of clusters if not specified
        if n_clusters is None:
            # Use a reasonable number based on data size
            n_clusters = min(max(2, len(unique_skills) // 5), max_clusters)
        
        if n_clusters >= len(unique_skills):
            # Each skill gets its own cluster
            return {skill: i for i, skill in enumerate(unique_skills)}
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create mapping
        skill_clusters = {skill: int(cluster_id) for skill, cluster_id in zip(unique_skills, cluster_labels)}
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Skill clustering completed in {elapsed:.1f}ms")
        
        return skill_clusters
        
    except Exception as e:
        logger.error(f"Error in skill clustering: {e}")
        # Fallback: assign all skills to cluster 0
        return {skill: 0 for skill in unique_skills}