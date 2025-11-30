import re
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Keywords for seniority detection
SENIORITY_KEYWORDS = {
    "intern": ["intern", "internship", "interno", "practicante", "becario"],
    "junior": ["junior", "entry level", "entry-level", "entry", "trainee", "principiante", "inicial"],
    "mid": ["mid-level", "mid level", "intermediate", "intermedio", "medio"],
    "senior": ["senior", "sr", "sr.", "sénior", "experto", "experiencia"],
    "lead": ["lead", "team lead", "tech lead", "technical lead", "líder técnico"],
    "principal": ["principal", "principal engineer", "principal developer"],
    "staff": ["staff", "staff engineer", "staff developer"],
    "architect": ["architect", "software architect", "solution architect", "arquitecto"],
    "manager": ["manager", "engineering manager", "tech manager", "gerente", "jefe"],
    "director": ["director", "director of engineering", "director técnico"],
}

def detect_seniority(job_title: str, job_description: str = "") -> Dict[str, any]:
    """
    Detect seniority level from job title and description.
    
    Returns:
        dict with keys:
            - level: str (intern/junior/mid/senior/lead/principal/staff/architect/manager/director/unknown)
            - score: float (0-1, confidence score)
            - keywords_found: list of matched keywords
    """
    if not job_title:
        return {"level": "unknown", "score": 0.0, "keywords_found": []}
    
    text = f"{job_title} {job_description}".lower()
    
    # Count matches for each seniority level
    matches = {}
    all_keywords_found = []
    
    for level, keywords in SENIORITY_KEYWORDS.items():
        count = 0
        found_keywords = []
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
                found_keywords.append(keyword)
        
        if count > 0:
            matches[level] = count
            all_keywords_found.extend(found_keywords)
    
    if not matches:
        return {"level": "unknown", "score": 0.0, "keywords_found": []}
    
    # Get the level with most matches
    best_level = max(matches.items(), key=lambda x: x[1])
    level, match_count = best_level
    
    # Calculate confidence score (normalized by number of keywords for that level)
    total_keywords_for_level = len(SENIORITY_KEYWORDS[level])
    score = min(match_count / total_keywords_for_level, 1.0)
    
    return {
        "level": level,
        "score": score,
        "keywords_found": list(set(all_keywords_found))
    }

def get_seniority_score(job_title: str, job_description: str = "") -> float:
    """
    Get a numeric seniority score (0-10) where:
    0-1: intern
    2-3: junior
    4-5: mid
    6-7: senior
    8-9: lead/principal/staff
    10: architect/manager/director
    """
    result = detect_seniority(job_title, job_description)
    level = result["level"]
    
    score_map = {
        "intern": 1,
        "junior": 2.5,
        "mid": 5,
        "senior": 7,
        "lead": 8,
        "principal": 8.5,
        "staff": 8.5,
        "architect": 9,
        "manager": 9.5,
        "director": 10,
        "unknown": 5  # Default to mid if unknown
    }
    
    base_score = score_map.get(level, 5)
    # Adjust by confidence
    adjusted_score = base_score * (0.5 + 0.5 * result["score"])
    
    return min(adjusted_score, 10.0)

