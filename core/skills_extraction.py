import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher
import logging

from .config import TAXONOMY_PATH

logger = logging.getLogger(__name__)

# --- Load NLP model ---
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download xx_ent_wiki_sm")

# --- Load taxonomy with synonyms ---
taxonomy_df = pd.read_csv(TAXONOMY_PATH)
skills_list = taxonomy_df["skill"].tolist()

# Build synonym mapping: synonym -> canonical skill name
synonym_to_skill = {}
skill_patterns = []

for _, row in taxonomy_df.iterrows():
    skill = row["skill"]
    synonyms_str = row.get("synonyms", "")
    
    # Add canonical skill name
    skill_patterns.append(skill)
    synonym_to_skill[skill.lower()] = skill
    
    # Add synonyms if they exist
    if pd.notna(synonyms_str) and synonyms_str:
        synonyms = [s.strip() for s in str(synonyms_str).split("|") if s.strip()]
        for synonym in synonyms:
            skill_patterns.append(synonym)
            synonym_to_skill[synonym.lower()] = skill

# PhraseMatcher with all patterns (skills + synonyms)
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in skill_patterns]
matcher.add("SKILLS", patterns)

def clean_html(text):
    """Clean HTML tags from text."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def normalize_skill(skill_text):
    """Normalize detected skill to canonical name using synonym mapping."""
    skill_lower = skill_text.lower()
    return synonym_to_skill.get(skill_lower, skill_text.title())

def extract_skills(description):
    """
    Extract skills from job description using NLP and synonym matching.
    Returns list of canonical skill names (normalized).
    """
    if not description:
        return []
    
    try:
        doc = nlp(description)
        matches = matcher(doc)
        
        # Extract matched text and normalize to canonical skill names
        found_skills = set()
        for _, start, end in matches:
            matched_text = doc[start:end].text
            canonical_skill = normalize_skill(matched_text)
            found_skills.add(canonical_skill)
        
        return sorted(list(found_skills))
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []

def get_skills_by_category():
    """Return skills grouped by category."""
    return taxonomy_df.groupby("category")["skill"].apply(list).to_dict()

def get_all_skills():
    """Return list of all canonical skill names."""
    return skills_list
