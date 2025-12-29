"""
Welcome screen for the SkillGap application
"""

import streamlit as st


def render_welcome_screen():
    """
    Render the welcome screen when no data is available.
    """
    st.markdown("### Welcome to SkillGap")
    st.markdown(
        "Discover how well you match real job offers, what skills you're missing, "
        "and how to prioritize your upskilling roadmap."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="recommendation-card">
                <h4>1. Search Jobs</h4>
                <p>Pick a <strong>role</strong>, <strong>location</strong>, country and search radius in the sidebar.</p>
                <p>Optionally, start from an <strong>example scenario</strong> to see instant results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="recommendation-card">
                <h4>2. Add Your Skills</h4>
                <p>Select the skills that best describe you from the list in the sidebar.</p>
                <p>You can also type <strong>custom skills</strong> or upload your <strong>CV (PDF)</strong> to detect skills automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="recommendation-card">
                <h4>3. Analyze & Improve</h4>
                <p>Explore the tabs to see your <strong>skill gap</strong>, best matching jobs, and a learning roadmap.</p>
                <p>Enable <strong>Graph Analysis</strong> for network insights on skills.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown(
        "**Get started:** choose a role and location in the sidebar, "
        "optionally load an example scenario, then click **Search Jobs**."
        " *Note: The application focus on tech roles. The application is still in development and some features may not work as expected.*"
    )

