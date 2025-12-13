import logging
import streamlit as st
import pandas as pd

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from core.api_client import load_or_fetch_jobs
from core.skills_extraction import (
    clean_html,
    extract_skills,
    extract_custom_skills,
    skills_list,
    detect_seniority,
    get_best_apply_link,
)
from core.analysis import compute_skill_gap, cluster_skills_dynamic
from ui.components import render_skill_levels_ui

logger = logging.getLogger(__name__)


def render_search_parameters():
    """
    Render job search parameters in the sidebar.
    
    Returns:
        tuple: (role, location, country, date_posted, work_from_home, 
                employment_types_str, job_requirements_str, radius_val)
    """
    st.sidebar.header("Job Search Parameters")
    
    role = st.sidebar.text_input(
        "Role",
        value="",
        placeholder="e.g., data analyst, marketing manager, nurse, teacher..."
    )
    location = st.sidebar.text_input("Location", "Madrid")
    country = st.sidebar.text_input("Country code", "es")
    
    # Quick UX hint for new users
    with st.sidebar.expander("Quick tips", expanded=False):
        st.markdown(
            "- **Step 1**: Enter a role and location\n"
            "- **Step 2**: Add some of your skills\n"
            "- **Step 3**: Click **Search Jobs** and explore the tabs"
        )
    
    date_posted = st.sidebar.selectbox(
        "Posted date",
        ["all", "today", "3days", "week", "month"]
    )
    remote = st.sidebar.checkbox("Only remote?")
    work_from_home = True if remote else None
    
    employment_types = st.sidebar.multiselect(
        "Employment Type",
        options=["FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"],
        default=[]
    )
    employment_types_str = ",".join(employment_types) if employment_types else None
    
    job_requirements = st.sidebar.multiselect(
        "Job Requirements",
        options=["under_3_years_experience", "more_than_3_years_experience", "no_experience", "no_degree"],
        default=[]
    )
    job_requirements_str = ",".join(job_requirements) if job_requirements else None
    
    # Interactive radius selection
    st.sidebar.subheader("Search Radius")
    radius_mode = st.sidebar.radio(
        "How wide do you want to search?",
        ["Only this city", "Up to 20 km", "Up to 50 km", "Custom"],
        index=0,
    )
    
    if radius_mode == "Only this city":
        radius_val = None  # API will use just the city
    elif radius_mode == "Up to 20 km":
        radius_val = 20.0
    elif radius_mode == "Up to 50 km":
        radius_val = 50.0
    else:
        custom_radius = st.sidebar.slider(
            "Custom radius (km)", min_value=0, max_value=200, value=20, step=5
        )
        radius_val = float(custom_radius) if custom_radius > 0 else None
    
    st.sidebar.caption(
        "Tip: smaller radius focuses on your city; larger radius explores nearby cities "
        "or relocation opportunities."
    )
    
    return (
        role,
        location,
        country,
        date_posted,
        work_from_home,
        employment_types_str,
        job_requirements_str,
        radius_val,
    )


def render_skills_input():
    """
    Render skills input section in the sidebar.
    
    Returns:
        tuple: (all_user_skills, custom_skills)
    """
    st.sidebar.header("Your Skills")
    st.sidebar.caption("Select skills from any category: technical, soft skills, languages, tools, etc.")
    user_skills = st.sidebar.multiselect(
        "Select your skills",
        options=skills_list,
        help="Choose from technical skills, soft skills, languages, design tools, and more"
    )
    
    # Custom skills input (to allow any kind of skill)
    st.sidebar.subheader("Add Custom Skills (optional)")
    custom_skills_input = st.sidebar.text_input(
        "Enter custom skills (comma-separated)",
        placeholder="e.g., Customer Service, Sales, Photography",
        help="Add skills that are not in the list above",
    )
    
    # Parse custom skills
    custom_skills = []
    if custom_skills_input:
        custom_skills = [s.strip() for s in custom_skills_input.split(",") if s.strip()]
    
    # Optional CV upload to auto-detect skills
    st.sidebar.subheader("Or upload your CV (PDF)")
    cv_file = st.sidebar.file_uploader(
        "Upload your CV (PDF only)",
        type=["pdf"],
        help="We will extract skills automatically from your CV using the same taxonomy.",
    )
    
    cv_skills = []
    if cv_file is not None:
        if PyPDF2 is None:
            st.sidebar.error(
                "PyPDF2 is not installed. Run `pip install PyPDF2` to enable CV parsing."
            )
        else:
            try:
                reader = PyPDF2.PdfReader(cv_file)
                text_chunks = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
                cv_text = "\n".join(text_chunks)
                if cv_text.strip():
                    cv_skills = extract_skills(cv_text)
                    if cv_skills:
                        st.sidebar.caption(
                            f"Detected **{len(cv_skills)} skills** in your CV based on the taxonomy."
                        )
                        # Showing a short preview so the user can decide what to add/adjust manually
                        preview = ", ".join(sorted(cv_skills)[:10])
                        st.sidebar.markdown(
                            f"_Example skills from CV_: {preview}"
                            + (" …" if len(cv_skills) > 10 else "")
                        )
                        with st.sidebar.expander("Show all detected CV skills", expanded=False):
                            st.markdown(", ".join(sorted(cv_skills)))
                        st.sidebar.caption(
                            "You can complement or correct this list using the selector and custom skills above."
                        )
                    else:
                        st.sidebar.caption(
                            "No skills from the taxonomy were detected in your CV text."
                        )
                else:
                    st.sidebar.caption("Could not read text from the uploaded PDF.")
            except Exception as e:  # pragma: no cover - runtime only
                st.sidebar.error(f"Error reading CV PDF: {e}")
    
    # Combine selected, custom and CV skills (unique)
    all_user_skills = sorted(set(list(user_skills) + custom_skills + cv_skills))
    
    if not all_user_skills:
        st.sidebar.info(
            "Add some skills above so the app can calculate your skill gap and "
            "show personalized recommendations."
        )
    
    return all_user_skills, custom_skills


def render_filters(df):
    """
    Render post-fetch filters in the sidebar.
    
    Args:
        df: DataFrame with the jobs
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    st.sidebar.header("Filter Results")
    
    # Filter by seniority
    if "seniority" in df.columns and df["seniority"].nunique() > 1:
        seniority_filter = st.sidebar.multiselect(
            "Seniority Level",
            options=sorted(df["seniority"].unique()),
            default=sorted(df["seniority"].unique())
        )
        if seniority_filter:
            df = df[df["seniority"].isin(seniority_filter)]
    
    # Filter by city
    if "city" in df.columns and df["city"].nunique() > 1:
        city_filter = st.sidebar.multiselect(
            "City",
            options=sorted([c for c in df["city"].unique() if pd.notna(c)]),
            default=sorted([c for c in df["city"].unique() if pd.notna(c)])
        )
        if city_filter:
            df = df[df["city"].isin(city_filter)]
    
    # Filter by match ratio
    min_match = st.sidebar.slider(
        "Minimum Match Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    df = df[df["match_ratio"] >= min_match]
    
    return df


def process_job_search(
    role,
    location,
    country,
    date_posted,
    work_from_home,
    employment_types_str,
    job_requirements_str,
    radius_val,
    all_user_skills,
    custom_skills,
    skill_levels,
):
    """
    Process job search and return results.
    
    Args:
        role: Job role to search for
        location: Location to search in
        country: Country code
        date_posted: Date posted filter
        work_from_home: Remote work filter
        employment_types_str: Employment types filter
        job_requirements_str: Job requirements filter
        radius_val: Search radius value
        all_user_skills: List of user skills
        custom_skills: List of custom skills
        skill_levels: Dictionary of skill levels
    
    Returns:
        tuple: (df, missing, all_skills_flat, skill_clusters, complete_skill_levels)
    """
    # Reset pagination when starting a new search
    st.session_state.job_matches_page = 1
    
    # Basic validation to avoid confusing empty searches
    if not role.strip():
        st.warning("Please enter at least a role before searching for jobs.")
        st.stop()
    
    if not location.strip():
        st.warning("Please enter a location before searching for jobs.")
        st.stop()
    
    with st.spinner("Fetching and analyzing jobs..."):
        try:
            data = load_or_fetch_jobs(
                role,
                location,
                country,
                date_posted=date_posted,
                work_from_home=work_from_home,
                employment_types=employment_types_str,
                job_requirements=job_requirements_str,
                radius=radius_val,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching jobs: {error_msg}")
            if "502" in error_msg:
                st.error(
                    "The external job API is temporarily unavailable (502 Bad Gateway). "
                    "Please try again in a few minutes or adjust the search filters "
                    "(role, location, radius)."
                )
            else:
                st.error(f"Error fetching jobs from API: {error_msg}")
            st.stop()
        
        job_results = data.get("data", [])
        
        if not job_results:
            st.warning("No jobs found with the current search parameters.")
            st.stop()
        
        rows = []
        for job in job_results:
            desc = clean_html(job.get("job_description", ""))
            skills = extract_skills(desc)
            
            # Also search for custom skills in the description
            if custom_skills:
                found_custom_skills = extract_custom_skills(desc, custom_skills)
                skills.extend(found_custom_skills)
                # Remove duplicates while preserving order
                skills = list(dict.fromkeys(skills))
            
            title = job.get("job_title", "")
            seniority = detect_seniority(title, desc)
            apply_link = get_best_apply_link(job)
            
            rows.append({
                "job_id": job.get("job_id"),
                "title": title,
                "company": job.get("employer_name"),
                "city": job.get("job_city"),
                "seniority": seniority,
                "skills_detected": skills,
                "apply_link": apply_link,
            })
        
        rows, missing = compute_skill_gap(rows, all_user_skills, skill_levels)
        
        if not rows:
            st.warning("No jobs found with the current filters.")
            st.stop()
        
        df = pd.DataFrame(rows)
        
        # Apply post-fetch filters
        df = render_filters(df)
        
        if len(df) == 0:
            st.warning("No jobs match the selected filters.")
            st.stop()
        
        df = df.sort_values("match_ratio", ascending=False)
        
        # Prepare data for all tabs
        all_skills_flat = []
        for skills in df["skills_detected"]:
            all_skills_flat.extend(skills if isinstance(skills, list) else [])
        
        # Dynamic skill clustering using embeddings
        unique_skills = sorted(list(set(all_skills_flat)))
        skill_clusters = cluster_skills_dynamic(unique_skills, max_clusters=8)
        
        # Ensure all skills have a level (default to "Expert" if not set)
        complete_skill_levels = {}
        for skill in all_user_skills:
            # Get from skill_levels dict or from session state or default to "Expert"
            session_key = f"skill_level_{skill}"
            if skill in skill_levels:
                complete_skill_levels[skill] = skill_levels[skill]
            elif session_key in st.session_state:
                complete_skill_levels[skill] = st.session_state[session_key]
            else:
                complete_skill_levels[skill] = "Expert"
                st.session_state[session_key] = "Expert"
        
        return df, missing, all_skills_flat, skill_clusters, complete_skill_levels

