import streamlit as st
from pathlib import Path


def render_skill_levels_ui(all_user_skills, skill_levels, key_suffix=""):
    """
    Render the skill levels UI in the sidebar.
    
    Args:
        all_user_skills: List of user skills
        skill_levels: Dictionary to store skill levels
        key_suffix: Optional suffix to add to widget keys for uniqueness
    
    Returns:
        dict: Updated skill_levels dictionary
    """
    if not all_user_skills:
        return skill_levels
    
    st.sidebar.divider()
    st.sidebar.subheader("Skill Levels (Optional)")
    st.sidebar.caption("Set your proficiency level for each skill. Default: Expert if not specified.")
    
    # Level options
    level_options = ["Basic", "Intermediate", "Advanced", "Expert"]
    
    # Use expander to keep sidebar clean
    with st.sidebar.expander("Set Skill Levels", expanded=False):
        for skill in all_user_skills:
            # Session state key (shared across renders)
            session_key = f"skill_level_{skill}"
            # Widget key (unique per render location)
            widget_key = f"skill_level_{skill}{key_suffix}"
            
            if session_key not in st.session_state:
                st.session_state[session_key] = "Expert"
            
            selected_level = st.selectbox(
                f"{skill}",
                options=level_options,
                index=level_options.index(st.session_state[session_key]),
                key=widget_key,
                help=f"Select your proficiency level for {skill}"
            )
            # Update session state
            st.session_state[session_key] = selected_level
            skill_levels[skill] = selected_level
    
    return skill_levels


def render_header():
    """
    Render the main application header.
    """
    st.markdown('<h1 class="main-header">SKILLGAP</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Skill Gap Analysis for Job Seekers</p>', unsafe_allow_html=True)


def render_footer():
    """
    Render the application footer with acknowledgments and author information.
    """
    base_dir = Path(__file__).resolve().parent.parent
    logo_path = base_dir / "assets" / "openwebninja.png"
    
    # Check if logo exists
    if logo_path.exists():
        try:
            import base64
            with open(logo_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                logo_html = f' <a href="https://openwebninja.com" target="_blank" class="footer-logo-link"> <img src="data:image/png;base64,{img_data}" alt="OpenWeb Ninja" class="footer-logo"/> <span class="footer-logo-text">OpenWeb Ninja</span> </a>'
        except Exception:
            logo_html = '<img src="data:image/png;base64,{img_data}" alt="OpenWeb Ninja" class="footer-logo" <span class="footer-logo-text">OpenWeb Ninja</span>'
    else:
        logo_html = '<img src="data:image/png;base64,{img_data}" alt="OpenWeb Ninja" class="footer-logo" <span class="footer-logo-text">OpenWeb Ninja</span>'


    # Cargar iconos una sola vez
    linkedin_image_path = base_dir / "assets" / "linkedin.png"
    github_image_path = base_dir / "assets" / "github.png"
    
    with open(linkedin_image_path, "rb") as linkedin_image:
        linkedin_image_data = base64.b64encode(linkedin_image.read()).decode()
    
    with open(github_image_path, "rb") as github_image:
        github_image_data = base64.b64encode(github_image.read()).decode()
    
    
    # Definir desarrolladores con sus enlaces
    developers = [
        {
            "name": "Carolina Lopez de la Madriz",
            "github": "https://github.com/carolinalopezdelamadriz",
            "linkedin": "https://www.linkedin.com/in/carolinalopezdelamadriz/"
        },
        {
            "name": "Emma Rodriguez Hervas",
            "github": "https://github.com/emmarhv",
            "linkedin": "https://www.linkedin.com/in/emma-rodriguez-hervas/"
        },
        {
            "name": "Álvaro Martin Ruiz",
            "github": "https://github.com/alvaromartinruiz",
            "linkedin": "https://www.linkedin.com/in/alvaro-martin-ruiz-engineering/ "
        },
        {
            "name": "Iker Rosales Saiz",
            "github": "https://github.com/ikerosales",
            "linkedin": "https://www.linkedin.com/in/iker-rosales-saiz-49218531b/"
        }
    ]

    developers_html = ''
    for dev in developers:
        developers_html += f'<div class="developer-card"> <span class="developer-name">{dev['name']}</span><div class="social-links"><a href="{dev['github']}" target="_blank" rel="noopener noreferrer" class="social-link" title="GitHub"><img src="data:image/png;base64,{github_image_data}" alt="GitHub" class="social-icon"/></a><a href="{dev['linkedin']}" target="_blank" rel="noopener noreferrer" class="social-link" title="LinkedIn"><img src="data:image/png;base64,{linkedin_image_data}" alt="LinkedIn" class="social-icon"/></a></div></div>'
    footer_html = f"""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-section">
                <p class="footer-text">Powered by job data from</p>
                <div class="footer-logo-container">
                    {logo_html}
                </div>
            </div>
            <div class="footer-divider"></div>
            <div class="footer-section">
                <p class="footer-authors">Developed by</p>
                <div class="developers-container">
                    {developers_html}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

