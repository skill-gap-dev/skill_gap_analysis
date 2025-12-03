from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TAXONOMY_PATH = DATA_DIR / "taxonomy_skills.csv"

MAX_NUM_PAGES = 3  # evitar gastar cuota

VALID_DATE_POSTED = {"all", "today", "3days", "week", "month"}
VALID_EMPLOYMENT_TYPES = {"FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"}
VALID_JOB_REQUIREMENTS = {
    "under_3_years_experience",
    "more_than_3_years_experience",
    "no_experience",
    "no_degree",
}
