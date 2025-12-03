import logging
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher
import re

from .config import TAXONOMY_PATH

# Setup logging
logger = logging.getLogger(__name__)

# --- Load NLP model ---
try:
    nlp = spacy.load("xx_ent_wiki_sm")
    logger.info("spaCy model loaded successfully")
except OSError as e:
    logger.error(f"spaCy model not found: {e}")
    raise RuntimeError("Run: python -m spacy download xx_ent_wiki_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    raise

# --- Load taxonomy with synonyms ---
try:
    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    logger.info(f"Loaded taxonomy with {len(taxonomy_df)} skills from {TAXONOMY_PATH}")
except FileNotFoundError:
    logger.error(f"Taxonomy file not found: {TAXONOMY_PATH}")
    raise
except Exception as e:
    logger.error(f"Error loading taxonomy: {e}")
    raise

def normalize_token(x: str) -> str:
    if not x:
        return ""
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9+.#]+", " ", x)         # remove weird chars
    x = re.sub(r"s$", "", x)                    # remove plural "s"
    x = x.replace("-", " ")                     # machine-learning → machine learning
    x = x.replace("_", " ")
    return x

# Create mapping from synonyms to main skill name
synonym_to_skill = {}
all_patterns = []

for _, row in taxonomy_df.iterrows():
    main_skill = row["skill"]
    synonyms_str = row.get("synonyms", "")
    
    # Add main skill to patterns
    norm = normalize_token(main_skill)
    all_patterns.append(main_skill)
    synonym_to_skill[norm] = main_skill
    
    # Add synonyms to patterns and mapping
    if pd.notna(synonyms_str) and synonyms_str.strip():
        synonyms = [s.strip() for s in synonyms_str.split("|")]
        for syn in synonyms:
            norm = normalize_token(syn)
            if norm not in synonym_to_skill:
                all_patterns.append(syn)
                synonym_to_skill[norm] = main_skill

# PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in all_patterns]
matcher.add("SKILLS", patterns)

# Keep original skills_list for backward compatibility
skills_list = taxonomy_df["skill"].tolist()

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ") if text else ""

def extract_skills(description):
    """
    Extract skills from job description using NLP matching.
    Returns normalized skill names (main skill, not synonyms).
    """
    doc = nlp(description)
    matches = matcher(doc)
    
    # Map found matches to main skill names
    found_skills = set()
    for _, start, end in matches:
        matched_text = doc[start:end].text
        # Map synonym to main skill
        main_skill = synonym_to_skill.get(normalize_token(matched_text))
        if main_skill:
            found_skills.add(main_skill)
    
    return sorted(list(found_skills))


def extract_custom_skills(description, custom_skills):
    """
    Extract custom skills from job description using text matching.
    
    Args:
        description: Job description text
        custom_skills: List of custom skill names to search for
    
    Returns:
        List of custom skills found in the description
    """
    if not custom_skills or not description:
        return []
    
    description_lower = description.lower()
    found_custom_skills = []
    
    for skill in custom_skills:
        if not skill or not skill.strip():
            continue
        
        # Normalize the skill name for matching
        skill_normalized = normalize_token(skill)
        skill_lower = skill.lower()
        
        # Check if skill appears in description (case-insensitive)
        # Use word boundaries to avoid partial matches
        # For example, "Python" should match "Python" but not "Pythonic"
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, description_lower):
            found_custom_skills.append(skill)
    
    return found_custom_skills

def detect_seniority(job_title, job_description=""):
    """
    Detect seniority level from job title and description.
    Returns: 'junior', 'mid', 'senior', or 'unknown'
    """
    if not job_title:
        job_title = ""
    if not job_description:
        job_description = ""
    
    # Combine text and convert to lowercase
    text = (job_title + " " + job_description).lower()
    
    # Keywords for each level
    junior_keywords = [
        "junior", "júnior", "entry", "entry-level", "entry level",
        "intern", "internship", "trainee", "graduate", "new grad",
        "associate", "asociado", "primeros pasos"
    ]
    
    senior_keywords = [
        "senior", "sénior", "sr.", "sr ", "principal",
        "staff", "director", "head of",
        "manager", "managing", "chief", "4+ years", "5+ years",
        "6+ years", "7+ years", "8+ years", "9+ years", "10+ years"
    ]
    
    mid_keywords = [
        "mid", "mid-level", "mid level", "intermediate", "medio",
        "experienced", "2+ years", "3+ years", "1-2 years", "2-3 years"
    ]
    
    # Count matches
    junior_count = sum(1 for keyword in junior_keywords if keyword in text)
    senior_count = sum(1 for keyword in senior_keywords if keyword in text)
    mid_count = sum(1 for keyword in mid_keywords if keyword in text)
    
    # Determine level (priority: mid > junior > senior)
    if mid_count > 0:
        return "mid"
    elif junior_count > 0:
        return "junior"
    elif senior_count > 0:
        return "senior"
    else:
        return "unknown"


def get_best_apply_link(job):
    # 1) Search LinkedIn in apply_options field
    apply_opts = job.get("apply_options", [])
    for opt in apply_opts:
        if opt.get("publisher", "").lower() == "linkedin":
            return opt.get("apply_link")

    # 2) If not LinkedIn, use job_apply_link
    if job.get("job_apply_link"):
        return job["job_apply_link"]

    # If there is not job_apply_link, use first apply_option
    if apply_opts:
        return apply_opts[0].get("apply_link")

    # 4) Si no hay nada, devolver None
    return None