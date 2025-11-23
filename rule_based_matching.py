import os
import json
import csv
from collections import Counter
from textwrap import shorten
from pathlib import Path

import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup  # For cleaning HTML from descriptions
import spacy
from spacy.matcher import PhraseMatcher

# --- 1. ENVIRONMENT & NLP MODEL LOADING ---
load_dotenv()

# Legacy variables (Adzuna) not used anymore, but kept for future use if needed
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")

# JSearch / OpenWebNinja API key
API_KEY_JSEARCH = os.getenv("API_KEY_JSEARCH")

if not API_KEY_JSEARCH:
    raise RuntimeError("API_KEY_JSEARCH is not set in the .env file.")

# Load NLP spaCy model (small spanish) for rule-based skill extraction
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    print("Model 'xx_ent_wiki_sm' not found.\n"
        "Run: python -m spacy download xx_ent_wiki_sm\n"
        "or install via: pip install spacy && python -m spacy download xx_ent_wiki_sm")
    exit()

# Base folder for cached API responses and processed data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# To avoid burning the 200 requests/month, we keep num_pages small.
MAX_NUM_PAGES = 1  # up to 10 jobs per query in a single API call

# --- 2. SKILL TAXONOMY & MATCHER (HARDCODED PARA PROTOTIPO) ---
# In a more advanced version, this will come from an external taxonomy
# (ESCO, CSV file, database). For this prototype we hardcode a list.

# We normalize matching to lowercase to avoid duplicates.
taxonomy_skills = [
    "Python", "SQL", "R", "Excel", "Tableau", "Power BI", "Java", 
    "Machine Learning", "AWS", "Azure", "Spark", "Hadoop", 
    "Data Visualization", "Estadística", "Inglés", "Git", "Scrum",
    "Communication", "Teamwork", "NoSQL", "Pandas", "Numpy"
]

# Initialize PhraseMatcher 
matcher = PhraseMatcher(nlp.vocab, attr="LOWER") # 'LOWER' = case-insensitive
patterns = [nlp.make_doc(text) for text in taxonomy_skills]
matcher.add("SKILL_LIST", patterns)

# --- 3. HELPER FUNCTIONS ---

def clean_html(html_text):
    """Strip HTML tags from job description text."""
    if not html_text: return ""
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ")

def extract_skills(description_text):
    """Use spaCy + PhraseMatcher to detect which skills from the taxonomy appear in the job description."""
    doc = nlp(description_text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        # Original text that matched
        span = doc[start:end]
        found_skills.add(span.text.title()) # .title() just for nicer display (e.g. 'python' -> 'Python')
    
    return list(found_skills)

def sanitize_for_filename(value):
    """
    Create a safe file version with the parameters strings introduced by the user
    to use inside filenames.
    """
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_")

def build_query(role, location):
    """Creates text query for JSearch.
    According to the documentation of JSearch, the query format recommended is:
    <role> jobs in <location>
    """
    role_part = role.strip() or "data analyst"
    location_part = location.strip()
    if location_part:
        return f"{role_part} jobs in {location_part}"
    return f"{role_part} jobs"

def get_cache_paths(role, location):
    """Generate file paths for cached raw JSON and processed CSV based on role and location."""
    # WARNING - TODO: los nombres de archivo pueden ser muy largos si el rol o la localizacion son largos
    # WARNING - TODO: si cambiamos filtros pero no el rol/location, vamos a coger el fichero existente en vez hacer un nuevo request
    
    role_token = sanitize_for_filename(role or "role")
    loc_token = sanitize_for_filename(location or "location")
    raw_path = DATA_DIR / f"raw_jobs_{role_token}_{loc_token}.json"
    csv_path = DATA_DIR / f"processed_jobs_{role_token}_{loc_token}.csv"
    return raw_path, csv_path


# --- 4. API CALLING (WITH FILTERS) ---

# Valid values according to JSearch documentation
VALID_DATE_POSTED = {"all", "today", "3days", "week", "month"}

VALID_EMPLOYMENT_TYPES = {
    "FULLTIME",
    "PARTTIME",
    "CONTRACTOR",
    "INTERN",
}

VALID_JOB_REQUIREMENTS = {
    "under_3_years_experience",
    "more_than_3_years_experience",
    "no_experience",
    "no_degree",
}
def fetch_jobs_from_api(
        query, 
        country = "es", 
        num_pages=1, 
        date_posted="all", 
        work_from_home:bool | None = None, 
        employment_types:str | None = None,
        job_requirements: str | None = None,
        radius: float | None = None):
     
    """
    Call the JSearch /search endpoint and return the parsed JSON response.

    This function supports the main filters exposed in the documentation:
    WARNING: each call consumes JSearch monthly quota. This is why we later
    cache results to disk and reuse them.

    Parameters
    ----------
    query : str
        Free-form search query (e.g. "data analyst jobs in Madrid").
    country : str
        ISO 3166-1 alpha-2 country code, e.g. 'es', 'us', 'de'.
    num_pages : int
        Number of pages to retrieve. Each page can contain up to 10 jobs.
    date_posted : str 
        Time window filter for posting date.
        Options: 'all', 'today', '3days', 'week', 'month'.
    work_from_home : bool | None
        If True, only remote jobs. If None, do not filter.
    employment_types : str | None
        Comma-separated job types.
        Options: 'FULLTIME', 'PARTTIME', 'CONTRACTOR', 'INTERN'.
    job_requirements : str | None
        Comma-separated experience/degree filters.
        Options: 'under_3_years_experience', 'more_than_3_years_experience',
                 'no_experience', 'no_degree'
    radius : float | None
        Max distance (km) from the location included in the `query`.

    Returns
    -------
    dict
        Parsed JSON response from the API.
    """   

    # Sanitization and validation of input parameters
    country = country.lower()
    if len(country) != 2 or not country.isalpha():
        print(f"Invalid country '{country}'. Falling back to 'ES'.")
        country = "es"

    if date_posted not in VALID_DATE_POSTED:
        print(f"Invalid date_posted '{date_posted}'. Falling back to 'all'."
              "Allowed values: all, today, 3days, week, month. Using 'all'.")
        date_posted = "all"
    
    try:
        num_pages = int(num_pages)
    except ValueError:
        print(f"Invalid num_pages '{num_pages}'. Falling back to 1.")
        num_pages = 1
    
    #  employment_types: normalize to comma-separated valid values or None
    normalized_employment_types = None
    if employment_types:
        raw_tokens = employment_types.split(",")
        cleaned = []
        for token in raw_tokens:
            token = token.strip().upper()
            if token in VALID_EMPLOYMENT_TYPES:
                cleaned.append(token)
            else:
                print(f"Warning: invalid employment_type '{token}' ignored."
                      f"Valid: {', '.join(sorted(VALID_EMPLOYMENT_TYPES))}")
        if cleaned:
            normalized_employment_types = ",".join(cleaned)
        else:
            normalized_employment_types = None

    # job_requirements: normalize to comma-separated valid values or None
    normalized_job_requirements = None
    if job_requirements:
        raw_tokens = job_requirements.split(",")
        cleaned = []
        for token in raw_tokens:
            token = token.strip().lower().replace(" ", "_")
            if token in VALID_JOB_REQUIREMENTS:
                cleaned.append(token)
            else:
                print(f"Warning: invalid job_requirement '{token}' ignored."
                      f"Valid: {', '.join(sorted(VALID_JOB_REQUIREMENTS))}")
        if cleaned:
            normalized_job_requirements = ",".join(cleaned)
        else:
            normalized_job_requirements = None
    
    # radius: ensure it's a float or None
    if radius is not None:
        try:
            radius = float(radius)
            if radius <= 0:
                print(f"Radius must be positive. Ignoring this filter.")
                radius = None
            
        except ValueError:
            print(f"Invalid radius '{radius}'. Ignoring this filter.")
            radius = None

    # 2) Build request parameters for JSearch
    url = "https://api.openwebninja.com/jsearch/search"
    headers = {"x-api-key": API_KEY_JSEARCH}

    params = {
        "query": query,
        "country": country,  # default to Spain if user leaves it empty
        "page": 1,
        # Clamp between 1 and MAX_NUM_PAGES to avoid huge calls in dev
        "num_pages": max(1, min(num_pages, MAX_NUM_PAGES)),
        "date_posted": date_posted,
        # To reduce payload size we request only the fields we actually use:

    }

    # Optional filters:
    if work_from_home is not None:
        # API expects "true"/"false" as lowercase strings
        params["work_from_home"] = str(work_from_home).lower()

    if normalized_employment_types:
        params["employment_types"] = normalized_employment_types

    if normalized_job_requirements:
        params["job_requirements"] = normalized_job_requirements

    if radius is not None:
        params["radius"] = radius

        # --- 3) Make request + robust fallback if 400 Bad Request ---
    response = requests.get(url, params=params, headers=headers, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        # If it's not a 400, just propagate the error
        if response.status_code != 400:
            raise

        print("JSearch returned 400 Bad Request. "
              "Trying to relax optional filters step by step...")

        # Try removing optional filters in this order:
        fallback_filters = ["radius","date_posted" ,"job_requirements", "employment_types", "work_from_home"]

        for key in fallback_filters:
            if key in params:
                print(f"   - Removing '{key}' and retrying...")
                params.pop(key)
                response = requests.get(url, params=params, headers=headers, timeout=30)
                try:
                    response.raise_for_status()
                    print("   → Request succeeded after removing filters.")
                    break
                except requests.HTTPError:
                    if response.status_code != 400:
                        # Different error: propagate it
                        raise
                    # Still 400: keep trying with the next filter
                    continue
        else:
            # Loop finished and we never broke -> still failing
            print(" All fallback attempts failed. Raising original error.")
            raise

    return response.json()

def load_or_fetch_jobs(role, location, country, num_pages=1, **filters):
    """
    This function:
      1) loads cached JSON results for (role, location) from disk, 
      OR
      2) calls JSearch, saves the JSON to disk, and returns it.

    This is the key piece for not exhausting the 200-requests/month quota.

    Parameters
    ----------
    role : str
        Job title for the search 
    location : str
        City/area for the search 
    country : str
        ISO country code (used by JSearch for strict country filter).
    num_pages : int
        Number of pages to retrieve via JSearch.
    **filters :
        Additional filters passed straight into `fetch_jobs_from_api`
        (date_posted, work_from_home, employment_types, job_requirements, radius).

    Returns
    -------
    dict
        JSON with JSearch results.
    """

    raw_path, _ = get_cache_paths(role, location)

    # 1. Load from cache if it exists
    if raw_path.exists():
        print(f"Loading cached results from: {raw_path}")
        with raw_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Otherwise, call the API (consumes quota) and cache the result
    print("No cache found → calling API (this uses your monthly quota!)")
    data = fetch_jobs_from_api(
        build_query(role, location),
        country,
        num_pages=num_pages,
        **filters
    )

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved API response to: {raw_path}")
    return data

def save_processed_to_csv(rows: list[dict], csv_path: Path):
    """
    Save the processed job rows (one dict per job) into a CSV file.

    Parameters
    ----------
    rows : list of dict
        Flattened job information + detected skills.
    csv_path : Path
        Path where the CSV will be written.
    """
    if not rows:
        print("No rows to save.")
        return

    fieldnames = rows[0].keys()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed data saved to: {csv_path}")

# --- 5. MAIN LOGIC---

def main():
    """
    Main function to run the rule-based skill gap analysis.
    
    Flow:
        1. Ask the user for role, location, and optional filters.
        2. Load cached jobs or call JSearch.
        3. Clean descriptions and extract skills.
        4. Print per-job summary.
        5. Print global skill demand summary.
        6. Save processed results to CSV.
    """
    print("=== SkillGap — Job Fetch & Skill Extraction ===")

    # --- Basic search parameters ---
    role = input("Role [default: data analyst]: ").strip() or "data analyst"
    location = input("Location [default: Madrid]: ").strip() or "Madrid"
    country = input("Country code (es/us/...) [default: es]: ").strip() or "es"

    # --- Optional filters ---
    date_posted = (
        input("Posted date [all/today/3days/week/month | ENTER=all]: ")
        .strip()
        or "all"
    )

    # Remote-only filter. If user answers 's', we set work_from_home=True.
    remote_raw = input("Only remote? [s/n, ENTER=n]: ").lower().strip()
    work_from_home = True if remote_raw == "s" else None
    
    # Employment types (comma-separated list to pass to the API)
    employment_types = (
        input(
            "Employment types (FULLTIME,PARTTIME,CONTRACTOR,INTERN) comma-separated "
            "[ENTER for any]: "
        )
        .strip()
        .upper()
        or None
    )

    # Additional requirements (experience, degree, etc.)
    job_requirements = (
        input(
            "Requirements (no_experience,no_degree,under_3_years_experience...) "
            "comma-separated [ENTER for any]: "
        )
        .strip()
        or None
    )

    # Radius in km around the location used in the query #TODO: no se si no meter radio significa que no hay limite o al contrario
    radius_raw = input("Radius in km [ENTER=no limit]: ").strip()
    radius = float(radius_raw) if radius_raw else None

    # --- LOAD OR FETCH JOBS ---
    data = load_or_fetch_jobs(
        role,
        location,
        country,
        num_pages=MAX_NUM_PAGES,
        date_posted=date_posted,
        work_from_home=work_from_home,
        employment_types=employment_types,
        job_requirements=job_requirements,
        radius=radius,
    )

    job_results = data.get("data", [])

    if not job_results:
        print("No jobs found.")
        return

    print(f"\nNumber of job postings retrieved: {len(job_results)}\n")

    all_skills = []
    rows = []

    # --- Per job: clean description, extract skills, print summary, build row ---
    for job in job_results:
        title = job.get("job_title")
        company = job.get("employer_name")
        city = job.get("job_city")

        raw_desc = job.get("job_description", "")
        desc = clean_html(raw_desc)
        skills = extract_skills(desc)

        all_skills.extend(skills)

        print("-" * 60)
        print(f"POSITION: {title} ({company}) — {city}")
        print("Skills:", ", ".join(skills) if skills else "No detected skills")
        print("Snippet:", shorten(desc, 150, placeholder="..."))

        rows.append({
            "job_id": job.get("job_id"),
            "title": title,
            "company": company,
            "city": city,
            "country": job.get("job_country"),
            "employment_type": job.get("job_employment_type"),
            "is_remote": job.get("job_is_remote"),
            "posted_at": job.get("job_posted_at"),
            "skills_detected": ", ".join(skills),
            "n_skills_detected": len(skills),
        })

    # --- Global skill demand summary (simple frequency analysis) ---
    print("\n" + "=" * 60)
    print("SKILL DEMAND SUMMARY")
    print("=" * 60)

    counts = Counter(all_skills)
    for skill, count in counts.most_common():
        pct = (count / len(job_results)) * 100
        print(f"{skill:<20} | {count} jobs ({pct:.0f}%)")

    # --- Save all processed jobs to CSV for later analysis/dashboards ---
    _, csv_path = get_cache_paths(role, location)
    save_processed_to_csv(rows, csv_path)


# ENTRY POINT 
if __name__ == "__main__":
    main()