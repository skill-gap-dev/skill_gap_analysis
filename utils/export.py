import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def export_to_csv(data: pd.DataFrame, filepath: str) -> bool:
    """Export DataFrame to CSV file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Exported data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return False

def export_to_json(data: List[Dict], filepath: str) -> bool:
    """Export list of dictionaries to JSON file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return False

def export_results_summary(jobs_df: pd.DataFrame, missing_skills: List[Dict],
                          user_skills: List[str], readiness_score: Dict,
                          filepath: str) -> bool:
    """
    Export comprehensive results summary to JSON.
    """
    summary = {
        "user_skills": user_skills,
        "readiness_score": readiness_score,
        "total_jobs_analyzed": len(jobs_df),
        "top_missing_skills": missing_skills[:10],
        "job_statistics": {
            "avg_match_ratio": float(jobs_df["match_ratio"].mean()) if "match_ratio" in jobs_df.columns else 0,
            "max_match_ratio": float(jobs_df["match_ratio"].max()) if "match_ratio" in jobs_df.columns else 0,
            "min_match_ratio": float(jobs_df["match_ratio"].min()) if "match_ratio" in jobs_df.columns else 0,
        },
        "top_matching_jobs": jobs_df.nlargest(5, "match_ratio")[
            ["title", "company", "match_ratio"]
        ].to_dict("records") if "match_ratio" in jobs_df.columns else []
    }
    
    return export_to_json([summary], filepath)

