import logging
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher

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

# Create mapping from synonyms to main skill name
synonym_to_skill = {}
all_patterns = []

for _, row in taxonomy_df.iterrows():
    main_skill = row["skill"]
    synonyms_str = row.get("synonyms", "")
    
    # Add main skill to patterns
    all_patterns.append(main_skill)
    synonym_to_skill[main_skill.lower()] = main_skill
    
    # Add synonyms to patterns and mapping
    if pd.notna(synonyms_str) and synonyms_str.strip():
        synonyms = [s.strip() for s in str(synonyms_str).split("|")]
        for synonym in synonyms:
            if synonym.lower() not in synonym_to_skill:
                all_patterns.append(synonym)
                synonym_to_skill[synonym.lower()] = main_skill

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
        matched_text = doc[start:end].text.lower()
        # Map synonym to main skill
        main_skill = synonym_to_skill.get(matched_text)
        if main_skill:
            found_skills.add(main_skill)
    
    return sorted(list(found_skills))


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
        "senior", "sénior", "sr.", "sr ", "lead", "líder", "principal",
        "staff", "expert", "architect", "director", "head of",
        "manager", "managing", "chief"
    ]
    
    mid_keywords = [
        "mid", "mid-level", "mid level", "intermediate", "medio",
        "experienced", "2+ years", "3+ years", "4+ years"
    ]
    
    # Count matches
    junior_count = sum(1 for keyword in junior_keywords if keyword in text)
    senior_count = sum(1 for keyword in senior_keywords if keyword in text)
    mid_count = sum(1 for keyword in mid_keywords if keyword in text)
    
    # Determine level (priority: senior > mid > junior)
    if senior_count > 0:
        return "senior"
    elif mid_count > 0:
        return "mid"
    elif junior_count > 0:
        return "junior"
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