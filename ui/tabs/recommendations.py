import streamlit as st
import pandas as pd
from core.skills_extraction import skills_list


def render_recommendations_tab(df, missing):
    """
    Render the Personalized Recommendations tab.
    
    Args:
        df: DataFrame with job data
        missing: List of missing skills
    """
    st.header("Personalized Recommendations")
    
    if missing:
        missing_df = pd.DataFrame(missing)
        
        # Priority recommendations
        st.subheader("Priority Skills to Develop")
        top_missing = missing_df.head(5)
        
        for idx, row in top_missing.iterrows():
            skill = row["skill"]
            count = row["count"]
            priority = row["priority"]
            
            # in percentage
            pct = (count / len(df)) * 100
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>{skill}</h4>
                <p><strong>Demand:</strong> Required in {count} jobs ({pct:.1f}% of all jobs)</p>
                <p><strong>Priority Level:</strong> {'High' if priority >= 5 else 'Medium' if priority >= 2 else 'Low'}</p>
                <p><strong>Why it matters:</strong> This skill appears frequently in job postings for your target role.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Skill development roadmap
        st.subheader("Skill Development Roadmap")
        
        # Group by category if possible
        skill_categories = {}
        for skill in top_missing["skill"].head(10):
            # Try to infer category from taxonomy
            category = "General"
            for s in skills_list:
                if s.lower() == skill.lower():
                    # Could add category lookup here
                    break
            skill_categories.setdefault(category, []).append(skill)
        
        roadmap_cols = st.columns(min(3, len(skill_categories)))
        for idx, (category, skills) in enumerate(skill_categories.items()):
            with roadmap_cols[idx % len(roadmap_cols)]:
                st.markdown(f"### {category}")
                for skill in skills[:5]:
                    st.markdown(f"- **{skill}**")
        
        # Action items
        st.subheader("Action Items")
        st.markdown("""
        **Immediate Actions (Next 30 Days):**
        - Focus on the top 3 missing skills with highest priority
        - Take online courses or tutorials for these skills
        - Practice through projects or exercises
        
        **Short-term Goals (Next 3 Months):**
        - Develop proficiency in at least 5 of the top missing skills
        - Build a portfolio showcasing these skills
        - Get certifications if available
        
        **Long-term Strategy (6-12 Months):**
        - Master the most critical skills for your target role
        - Stay updated with industry trends
        - Network with professionals in your field
        """)
        
        # Resources section
        st.subheader("Learning Resources")
        st.info("**Tip:** Search for online courses, tutorials, and certifications for the skills listed above. Platforms like Coursera, Udemy, LinkedIn Learning, and edX offer courses for most technical and soft skills.")
    
    else:
        st.success("""
        **Excellent!** You have all the required skills for the jobs in your search.
        
        **Next Steps:**
        - Focus on improving your proficiency levels in existing skills
        - Consider applying to positions with higher match ratios
        - Expand your search to more senior roles if applicable
        """)
    
    # General recommendations
    st.subheader("Career Development Tips")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Improve Your Profile:**
        - Update your resume with all relevant skills
        - Highlight your skill levels in your profile
        - Showcase projects demonstrating your abilities
        """)
    with col2:
        st.markdown("""
        **Job Search Strategy:**
        - Apply to jobs with match ratio ≥ 70%
        - Focus on roles matching your seniority level
        - Consider remote opportunities to expand options
        """)

