import streamlit as st


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

