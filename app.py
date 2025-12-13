"""
Main application file for SkillGap - Skill Gap Analysis for Job Seekers.

This file orchestrates the application by importing and using modular components.
"""

import logging
import streamlit as st

from ui.styles import apply_custom_css
from ui.components import render_header, render_skill_levels_ui
from ui.sidebar import (
    render_search_parameters,
    render_skills_input,
    process_job_search,
)
from ui.tabs.overview import render_overview_tab
from ui.tabs.skills_analysis import render_skills_analysis_tab
from ui.tabs.job_matches import render_job_matches_tab
from ui.tabs.recommendations import render_recommendations_tab
from ui.tabs.graph_analysis import render_graph_analysis_tab
from ui.welcome import render_welcome_screen

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config with modern styling
st.set_page_config(
    page_title="SkillGap",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

# Apply custom CSS
apply_custom_css()

# Render header
render_header()

# Render sidebar components
search_params = render_search_parameters()
all_user_skills, custom_skills = render_skills_input()

# Skill levels selection (optional) - only show if no search results yet
skill_levels = {}
# Only show skill level UI before search if we don't have results yet
if all_user_skills and 'df' not in st.session_state:
    skill_levels = render_skill_levels_ui(all_user_skills, skill_levels, key_suffix="_pre")

# Process job search when button is clicked
if st.sidebar.button("Search Jobs", type="primary", use_container_width=True):

    #
    (role, location, country, date_posted, work_from_home, 
    employment_types_str, job_requirements_str, radius_val) = search_params
    
    df, missing, all_skills_flat, skill_clusters, complete_skill_levels = process_job_search(
        role, location, country, date_posted, work_from_home, employment_types_str,
         job_requirements_str, radius_val, all_user_skills, 
         custom_skills, skill_levels)
    
    # Store in session state for tabs
    st.session_state.df = df
    st.session_state.all_user_skills = all_user_skills
    st.session_state.skill_levels = complete_skill_levels
    st.session_state.missing = missing
    st.session_state.all_skills_flat = all_skills_flat
    st.session_state.skill_clusters = skill_clusters

# Check if we have data to display
if 'df' in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df
    all_user_skills = st.session_state.all_user_skills
    skill_levels = st.session_state.skill_levels.copy() if st.session_state.skill_levels else {}
    missing = st.session_state.missing
    all_skills_flat = st.session_state.all_skills_flat
    skill_clusters = st.session_state.get('skill_clusters', {})
    
    # Show skill levels UI even after searching (so users can adjust)
    if all_user_skills:
        skill_levels = render_skill_levels_ui(all_user_skills, skill_levels, key_suffix="_post")
        # Update session state with current skill levels
        st.session_state.skill_levels = skill_levels
    
    st.sidebar.divider()
    st.sidebar.header("Advanced Analysis")
    enable_graph_analysis = st.sidebar.checkbox("Enable Graph Analysis", value=False)
    show_network_viz = st.sidebar.checkbox("Show Network Visualization", value=True) if enable_graph_analysis else False
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Skills Analysis",
        "Job Matches",
        "Recommendations",
        "Graph Analysis"
    ])
    
    with tab1:
        render_overview_tab(df, skill_levels, all_user_skills)
    
    with tab2:
        render_skills_analysis_tab(
            all_skills_flat,
            all_user_skills,
            skill_levels,
            missing,
            skill_clusters
        )
    
    with tab3:
        render_job_matches_tab(df)
    
    with tab4:
        render_recommendations_tab(df, missing)
    
    with tab5:
        render_graph_analysis_tab(
            df,
            all_user_skills,
            enable_graph_analysis,
            show_network_viz
        )

else:
    # Welcome screen
    render_welcome_screen()
