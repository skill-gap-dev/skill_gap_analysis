from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import logging

logger = logging.getLogger(__name__)

# Lazy loading of embedding model for performance
_embedding_model = None

def _get_embedding_model():
    """Lazy load the embedding model to avoid loading on import."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use fast, lightweight model optimized for speed
            # all-MiniLM-L6-v2 is fast and good quality
            _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    return _embedding_model

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
        if level is None:
            return 4  # Default to Expert (4) if not specified
        # If it's already a number (1-4), return it
        if isinstance(level, (int, float)):
            return int(level)
        # If it's a string, convert using mapping
        if isinstance(level, str):
            return level_name_to_numeric.get(level, 4)  # Default to Expert if unknown string
        return 4  # Fallback to Expert
    
    # Default numeric level is 4 (Expert) if skill is present but no level specified
    default_numeric_level = 4

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
        
        # Weighted match ratio: considers skill levels
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
            n_clusters = min(max(2, len(unique_skills) // 5), max_clusters, len(unique_skills))
        
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


def cluster_jobs(jobs_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Dynamically cluster jobs based on mean embeddings of their skills.
    
    Each job is represented as the mean embedding of all its skills, then
    KMeans clustering is applied to group similar jobs.
    
    Args:
        jobs_df: DataFrame with 'job_id' and 'skills_detected' columns
        n_clusters: Number of clusters (default: 4)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'cluster' column
    """
    start_time = time.time()
    
    if len(jobs_df) < 2:
        # Not enough jobs for clustering
        jobs_df = jobs_df.copy()
        jobs_df["cluster"] = 0
        return jobs_df
    
    # Collect all unique skills from all jobs
    all_skills_set = set()
    job_skills_list = []
    
    for _, row in jobs_df.iterrows():
        skills = row.get("skills_detected", [])
        
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        elif not isinstance(skills, list):
            skills = []
        
        # Filter out empty strings
        skills = [s for s in skills if s and s.strip()]
        all_skills_set.update(skills)
        job_skills_list.append(skills)
    
    if len(all_skills_set) == 0:
        # No skills found
        jobs_df = jobs_df.copy()
        jobs_df["cluster"] = 0
        return jobs_df
    
    try:
        model = _get_embedding_model()
        
        # Generate embeddings for all unique skills (batch processing for efficiency)
        unique_skills = sorted(list(all_skills_set))
        skill_embeddings = model.encode(unique_skills, show_progress_bar=False, convert_to_numpy=True)
        
        # Create mapping from skill to embedding
        skill_to_embedding = {skill: emb for skill, emb in zip(unique_skills, skill_embeddings)}
        
        # Represent each job as the mean embedding of its skills
        job_embeddings = []
        for skills in job_skills_list:
            if len(skills) == 0:
                # Job with no skills: use zero vector
                job_embeddings.append(np.zeros(skill_embeddings.shape[1]))
            else:
                # Get embeddings for all skills in this job
                skill_embs = [skill_to_embedding[skill] for skill in skills if skill in skill_to_embedding]
                if len(skill_embs) > 0:
                    # Mean of skill embeddings
                    job_emb = np.mean(skill_embs, axis=0)
                    job_embeddings.append(job_emb)
                else:
                    job_embeddings.append(np.zeros(skill_embeddings.shape[1]))
        
        job_embeddings = np.array(job_embeddings)
        
        # Determine number of clusters
        actual_n_clusters = min(n_clusters, len(jobs_df), max(2, len(jobs_df) // 3))
        
        if actual_n_clusters >= len(jobs_df):
            # Each job gets its own cluster
            clusters = np.arange(len(jobs_df))
        else:
            # Apply KMeans clustering
            kmeans = KMeans(
                n_clusters=actual_n_clusters, 
                random_state=random_state, 
                n_init=10, 
                max_iter=100
            )
            clusters = kmeans.fit_predict(job_embeddings)
        
        # Add cluster column
        result_df = jobs_df.copy()
        result_df["cluster"] = clusters
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Job clustering completed in {elapsed:.1f}ms")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in job clustering: {e}")
        # Fallback: assign all jobs to cluster 0
        result_df = jobs_df.copy()
        result_df["cluster"] = 0
        return result_df


def interpret_clusters(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpret clusters by finding most common skills in each cluster.
    
    Args:
        jobs_df: DataFrame with 'cluster' and 'skills_detected' columns
        
    Returns:
        DataFrame with cluster interpretations
    """
    cluster_summaries = []
    
    for cluster_id in sorted(jobs_df["cluster"].unique()):
        cluster_jobs = jobs_df[jobs_df["cluster"] == cluster_id]
        
        # Count skills in this cluster
        all_skills = []
        for skills in cluster_jobs["skills_detected"]:
            if isinstance(skills, list):
                all_skills.extend(skills)
            elif isinstance(skills, str):
                all_skills.extend([s.strip() for s in skills.split(",") if s.strip()])
        
        skill_freq = Counter(all_skills)
        top_skills = [skill for skill, _ in skill_freq.most_common(5)]
        
        cluster_summaries.append({
            "cluster": cluster_id,
            "num_jobs": len(cluster_jobs),
            "top_skills": ", ".join(top_skills),
            "avg_match_ratio": cluster_jobs["match_ratio"].mean() if "match_ratio" in cluster_jobs.columns else 0
        })
    
    return pd.DataFrame(cluster_summaries)
