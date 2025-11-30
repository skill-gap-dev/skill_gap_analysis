import os
import json
import requests
from pathlib import Path
import logging

from dotenv import load_dotenv
from .config import DATA_DIR, MAX_NUM_PAGES

logger = logging.getLogger(__name__)

load_dotenv()
API_KEY_JSEARCH = os.getenv("API_KEY_JSEARCH")

if not API_KEY_JSEARCH:
    logger.warning("API_KEY_JSEARCH missing in .env - API calls will fail")

def sanitize(value):
    return "".join(c.lower() if c.isalnum() else "_" for c in value)

def get_cache_paths(role, location):
    role_t = sanitize(role)
    loc_t = sanitize(location)
    raw = DATA_DIR / f"raw_jobs_{role_t}_{loc_t}.json"
    return raw

def fetch_from_api(query, country="es", **params):
    """
    Fetch jobs from JSearch API.
    
    Raises:
        RuntimeError: If API key is missing
        requests.RequestException: If API request fails
    """
    if not API_KEY_JSEARCH:
        raise RuntimeError("API_KEY_JSEARCH missing in .env file. Please set it up.")
    
    url = "https://api.openwebninja.com/jsearch/search"
    headers = {"x-api-key": API_KEY_JSEARCH}

    full_params = {
        "query": query,
        "country": country,
        "page": 1,
        "num_pages": MAX_NUM_PAGES,
        **params,
    }

    try:
        r = requests.get(url, params=full_params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        raise RuntimeError("API request timed out. Please try again later.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Invalid API key")
            raise RuntimeError("Invalid API key. Please check your API_KEY_JSEARCH in .env")
        elif e.response.status_code == 429:
            logger.error("API rate limit exceeded")
            raise RuntimeError("API rate limit exceeded. Please try again later.")
        else:
            logger.error(f"API HTTP error: {e}")
            raise RuntimeError(f"API error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise RuntimeError(f"Failed to fetch jobs: {e}")

def load_or_fetch_jobs(role, location, country="es", **filters):
    """
    Load jobs from cache or fetch from API.
    
    Returns:
        dict: API response with job data
    """
    raw_path = get_cache_paths(role, location)

    # Try to load from cache
    if raw_path.exists():
        try:
            with raw_path.open("r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.info(f"Loaded {len(cached_data.get('data', []))} jobs from cache")
                return cached_data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading cache file: {e}. Fetching from API...")

    # Fetch from API
    query = f"{role} jobs in {location}"
    try:
        data = fetch_from_api(query, country, **filters)
        
        # Save to cache
        try:
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(data.get('data', []))} jobs")
        except IOError as e:
            logger.warning(f"Could not write cache file: {e}")
        
        return data
    except RuntimeError:
        # Re-raise API errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load_or_fetch_jobs: {e}")
        raise RuntimeError(f"Failed to load or fetch jobs: {e}")
