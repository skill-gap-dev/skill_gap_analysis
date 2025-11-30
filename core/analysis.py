from collections import Counter
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

def compute_skill_gap(rows: List[Dict], user_skills: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute skill gap analysis between user skills and job requirements.
    
    Args:
        rows: List of job dictionaries with 'skills_detected' key
        user_skills: List of user's skills
    
    Returns:
        Tuple of (enhanced_rows, missing_skills)
    """
    user_skills = set(user_skills)
    
    # Calculate match metrics for each job
    for row in rows:
        job_sk = set(row.get("skills_detected", []))
        row["n_skills_job"] = len(job_sk)
        row["n_skills_user_has"] = len(job_sk & user_skills)
        row["n_skills_missing"] = len(job_sk - user_skills)
        
        # Match ratio
        row["match_ratio"] = (
            row["n_skills_user_has"] / row["n_skills_job"]
            if row["n_skills_job"] > 0 else 0
        )
        
        # Weighted match score (considers importance of skills)
        row["weighted_score"] = row["match_ratio"]  # Will be enhanced with graph analysis
    
    # Aggregate all skills across jobs
    all_skills = []
    for row in rows:
        all_skills.extend(row.get("skills_detected", []))
    
    freq = Counter(all_skills)
    total_jobs = len(rows)
    
    # Calculate missing skills with priority (frequency + percentage)
    missing = []
    for skill, count in freq.most_common():
        if skill not in user_skills:
            percentage = (count / total_jobs * 100) if total_jobs > 0 else 0
            missing.append({
                "skill": skill,
                "count": count,
                "percentage": round(percentage, 1),
                "priority": count  # Can be enhanced with centrality scores
            })
    
    return rows, missing

def compute_readiness_score(user_skills: List[str], missing_skills: List[Dict], 
                           total_jobs: int) -> Dict[str, float]:
    """
    Compute overall readiness score for the user.
    
    Returns:
        dict with readiness metrics
    """
    if total_jobs == 0:
        return {
            "readiness_score": 0.0,
            "coverage": 0.0,
            "top_missing_count": 0
        }
    
    # Coverage: percentage of unique skills user has
    all_required_skills = set()
    for skill_info in missing_skills:
        all_required_skills.add(skill_info["skill"])
    
    # This is simplified - in reality we'd need all skills from all jobs
    user_has_count = len(set(user_skills))
    total_unique_skills = user_has_count + len(all_required_skills)
    
    coverage = (user_has_count / total_unique_skills * 100) if total_unique_skills > 0 else 0
    
    # Readiness score (0-100)
    # Based on coverage and how many top skills are missing
    top_missing = len([s for s in missing_skills[:5]])  # Top 5 missing
    readiness_score = max(0, coverage - (top_missing * 10))
    
    return {
        "readiness_score": round(readiness_score, 1),
        "coverage": round(coverage, 1),
        "top_missing_count": top_missing
    }

def get_recommendations(user_skills: List[str], missing_skills: List[Dict], 
                       top_n: int = 5) -> List[Dict]:
    """
    Get intelligent recommendations for skills to learn.
    
    Args:
        user_skills: User's current skills
        missing_skills: List of missing skills with metadata
        top_n: Number of recommendations
    
    Returns:
        List of recommended skills with reasoning
    """
    recommendations = []
    
    for skill_info in missing_skills[:top_n]:
        skill = skill_info["skill"]
        count = skill_info["count"]
        percentage = skill_info.get("percentage", 0)
        
        # Generate recommendation message
        if percentage >= 50:
            priority = "High"
            reason = f"Required in {percentage}% of jobs"
        elif percentage >= 25:
            priority = "Medium"
            reason = f"Required in {percentage}% of jobs"
        else:
            priority = "Low"
            reason = f"Required in {percentage}% of jobs"
        
        recommendations.append({
            "skill": skill,
            "priority": priority,
            "reason": reason,
            "job_count": count,
            "percentage": percentage
        })
    
    return recommendations

def cluster_jobs(jobs_df: pd.DataFrame, n_clusters: int = 4, 
                 method: str = "kmeans") -> pd.DataFrame:
    """
    Cluster jobs based on their skill requirements.
    
    Args:
        jobs_df: DataFrame with 'skills_detected' column (list of skills per job)
        n_clusters: Number of clusters (for kmeans)
        method: 'kmeans' or 'dbscan'
    
    Returns:
        DataFrame with added 'cluster' column
    """
    if len(jobs_df) == 0:
        return jobs_df
    
    # Get all unique skills
    all_skills = set()
    for skills in jobs_df["skills_detected"]:
        all_skills.update(skills)
    all_skills = sorted(list(all_skills))
    
    if len(all_skills) == 0:
        jobs_df["cluster"] = 0
        return jobs_df
    
    # Create binary matrix: jobs x skills
    matrix = []
    for _, row in jobs_df.iterrows():
        skills = set(row["skills_detected"])
        vector = [1 if skill in skills else 0 for skill in all_skills]
        matrix.append(vector)
    
    X = np.array(matrix)
    
    # Apply clustering
    if method == "kmeans":
        if n_clusters > len(jobs_df):
            n_clusters = max(1, len(jobs_df) // 2)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(X, clusters)
            logger.info(f"KMeans clustering: {n_clusters} clusters, silhouette={silhouette:.3f}")
    
    elif method == "dbscan":
        # Standardize for DBSCAN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters = dbscan.fit_predict(X_scaled)
        
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        logger.info(f"DBSCAN clustering: {n_clusters_found} clusters found")
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    jobs_df = jobs_df.copy()
    jobs_df["cluster"] = clusters
    
    return jobs_df

def interpret_clusters(jobs_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Interpret clusters by finding top skills for each cluster.
    
    Returns:
        Dictionary mapping cluster_id -> {top_skills, job_count, description}
    """
    interpretations = {}
    
    for cluster_id in sorted(jobs_df["cluster"].unique()):
        cluster_jobs = jobs_df[jobs_df["cluster"] == cluster_id]
        
        # Count skills in this cluster
        skill_counts = Counter()
        for skills in cluster_jobs["skills_detected"]:
            skill_counts.update(skills)
        
        top_skills = [skill for skill, _ in skill_counts.most_common(5)]
        
        # Generate description
        if len(top_skills) > 0:
            description = f"Focus on: {', '.join(top_skills[:3])}"
        else:
            description = "General profile"
        
        interpretations[cluster_id] = {
            "top_skills": top_skills,
            "job_count": len(cluster_jobs),
            "description": description,
            "avg_match_ratio": cluster_jobs["match_ratio"].mean() if "match_ratio" in cluster_jobs else 0
        }
    
    return interpretations
