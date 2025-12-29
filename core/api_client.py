import logging
import streamlit as st
import requests
from .config import MAX_NUM_PAGES, API_KEY_JSEARCH

# Setup logging
logger = logging.getLogger(__name__)

if not API_KEY_JSEARCH:
    logger.error("API_KEY_JSEARCH missing in secrets")
    raise RuntimeError("API_KEY_JSEARCH missing in secrets")

def fetch_from_api(query, country_code="es", **params):
    """
    Fetch jobs from JSearch API.
    
    Args:
        query: Search query string
        country_code: Country code (default: "es")
        **params: Additional API parameters including:
            - date_posted: 'all', 'today', '3days', 'week', 'month'
            - work_from_home: bool or None
            - employment_types: comma-separated string (FULLTIME, PARTTIME, CONTRACTOR, INTERN)
            - job_requirements: comma-separated string (under_3_years_experience, more_than_3_years_experience, no_experience, no_degree)
            - radius: float or None
        
    Returns:
        JSON response from API
        
    Raises:
        requests.RequestException: If API request fails
    """
    from .config import VALID_DATE_POSTED, VALID_EMPLOYMENT_TYPES, VALID_JOB_REQUIREMENTS, API_URL

    # Validate and normalize parameters
    country_code = country_code.lower()
    if len(country_code) != 2 or not country_code.isalpha():
        logger.warning(f"Invalid country '{country_code}'. Falling back to 'es'.")
        country_code = "es"

    # Normalize date_posted
    date_posted = params.get("date_posted", "all")
    if date_posted not in VALID_DATE_POSTED:
        logger.warning(f"Invalid date_posted '{date_posted}'. Falling back to 'all'.")
        date_posted = "all"

    # Normalize employment_types
    employment_types = params.get("employment_types")
    normalized_employment_types = None
    if employment_types:
        raw_tokens = str(employment_types).split(",")
        cleaned = []
        for token in raw_tokens:
            token = token.strip().upper()
            if token in VALID_EMPLOYMENT_TYPES:
                cleaned.append(token)
            else:
                logger.warning(f"Invalid employment_type '{token}' ignored.")
        if cleaned:
            normalized_employment_types = ",".join(cleaned)

    # Normalize job_requirements
    job_requirements = params.get("job_requirements")
    normalized_job_requirements = None
    if job_requirements:
        raw_tokens = str(job_requirements).split(",")
        cleaned = []
        for token in raw_tokens:
            token = token.strip().lower().replace(" ", "_")
            if token in VALID_JOB_REQUIREMENTS:
                cleaned.append(token)
            else:
                logger.warning(f"Invalid job_requirement '{token}' ignored.")
        if cleaned:
            normalized_job_requirements = ",".join(cleaned)

    # Normalize radius
    radius = params.get("radius")
    if radius is not None:
        try:
            radius = float(radius)
            if radius <= 0:
                logger.warning("Radius must be positive. Ignoring this filter.")
                radius = None
        except (ValueError, TypeError):
            logger.warning(f"Invalid radius '{radius}'. Ignoring this filter.")
            radius = None

    headers = {"x-api-key": API_KEY_JSEARCH}

    full_params = {
        "query": query,
        "country": country_code,
        "page": 1, # each page is 10 results
        "num_pages": MAX_NUM_PAGES, #1 page -> 1 query requests, 2-10 pages -> 2 query requests
        "date_posted": date_posted,
    }

    # Add optional parameters
    work_from_home = params.get("work_from_home")
    if work_from_home is not None:
        # API expects "true"/"false" as lowercase strings
        full_params["work_from_home"] = str(work_from_home).lower()
    if normalized_employment_types:
        full_params["employment_types"] = normalized_employment_types
    if normalized_job_requirements:
        full_params["job_requirements"] = normalized_job_requirements
    if radius is not None:
        full_params["radius"] = radius

    try:
        logger.info(f"Fetching jobs from API: query='{query}', country='{country_code}'")
        r = requests.get(API_URL, params=full_params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        logger.info(f"Successfully fetched {len(data.get('data', []))} jobs")
        return data
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"API HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_or_fetch_jobs(role, location, country="es", **filters):
    """
    Fetch jobs from API with Streamlit caching.
    
    The results are cached for 1 hour to avoid excessive API calls.
    Cache is automatically invalidated after TTL expires.
    
    Args:
        role: Job role to search for
        location: Location to search in
        country: Country code (default: "es")
        **filters: Additional filters for API
        
    Returns:
        JSON data with job listings
    """
    query = f"{role} jobs in {location}"
    try:
        logger.info(f"Fetching jobs from API: role='{role}', location='{location}'")
        data = fetch_from_api(query, country, **filters)
        logger.info(f"Successfully fetched {len(data.get('data', []))} jobs")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch jobs: {str(e)}")
        raise
