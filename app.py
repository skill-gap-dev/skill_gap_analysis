import logging
import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

from core.api_client import load_or_fetch_jobs
from core.skills_extraction import clean_html, extract_skills, skills_list, detect_seniority, get_best_apply_link
from core.analysis import compute_skill_gap, cluster_jobs, interpret_clusters
from core.graph_analysis import (
    build_skill_cooccurrence_graph,
    compute_centralities,
    detect_communities,
    find_bridge_skills,
    plot_skill_network,
    get_skill_recommendations,
    get_skill_importance_scores,
    get_skill_paths
)

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

# Custom CSS for modern, professional corporate look
st.markdown("""
    <style>
    /* Main container dark theme */
    .main .block-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 10px;
    }
    
    .stApp {
        background: #0a0e27;
    }
    
    /* Header styling - large, white, bold */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #b8e994;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Headers in sidebar */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Text inputs and selects */
    .stTextInput > div > div > input, .stSelectbox > div > div > select {
        background-color: #2a2a3e;
        color: #ffffff;
        border: 1px solid #3a3a4e;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #1a1a2e;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 600;
        color: #a0a0a0;
        background-color: #2a2a3e;
        border: 1px solid #3a3a4e;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #ffffff;
        border: 2px solid #b8e994;
        box-shadow: 0 0 10px rgba(184, 233, 148, 0.3);
    }
    
    /* Headers in main content */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #b8e994 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #a0a0a0 !important;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1a1a2e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #b8e994;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }
    
    .recommendation-card h4 {
        color: #b8e994;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .recommendation-card p {
        color: #e0e0e0;
        margin: 0.5rem 0;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #2a2a3e;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #ffffff;
        border: 2px solid #b8e994;
        font-weight: 700;
        border-radius: 6px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border-color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(184, 233, 148, 0.4);
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #2a2a3e;
        border-left: 4px solid #b8e994;
        color: #ffffff;
    }
    
    .stSuccess {
        background-color: #1a3a1a;
        border-left: 4px solid #b8e994;
        color: #b8e994;
    }
    
    .stWarning {
        background-color: #3a2a1a;
        border-left: 4px solid #fbbf24;
        color: #fcd34d;
    }
    
    /* Text color */
    p, li, span, div {
        color: #e0e0e0;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Captions */
    .stCaption {
        color: #a0a0a0;
    }
    
    /* Dividers */
    hr {
        border-color: #3a3a4e;
    }
    
    /* Selectbox and multiselect */
    .stSelectbox label, .stMultiSelect label {
        color: #ffffff;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Header with corporate styling
st.markdown('<h1 class="main-header">SKILLGAP</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Skill Gap Analysis for Job Seekers</p>', unsafe_allow_html=True)

st.sidebar.header("Job Search Parameters")

role = st.sidebar.text_input("Role", value="", placeholder="e.g., data analyst, marketing manager, nurse, teacher...")
location = st.sidebar.text_input("Location", "Madrid")
country = st.sidebar.text_input("Country code", "es")

date_posted = st.sidebar.selectbox("Posted date", ["all", "today", "3days", "week", "month"])
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

radius = st.sidebar.number_input("Radius (km)", min_value=0.0, value=0.0)
radius_val = radius if radius > 0 else None

st.sidebar.header("Your Skills")
st.sidebar.caption("Select skills from any category: technical, soft skills, languages, tools, etc.")
user_skills = st.sidebar.multiselect("Select your skills", options=skills_list, help="Choose from technical skills, soft skills, languages, design tools, and more")

# Custom skills input
st.sidebar.subheader("Add Custom Skills")
custom_skills_input = st.sidebar.text_input(
    "Enter custom skills (comma-separated)",
    placeholder="e.g., Customer Service, Sales, Photography",
    help="Add skills that are not in the list above"
)

# Parse custom skills
custom_skills = []
if custom_skills_input:
    custom_skills = [s.strip() for s in custom_skills_input.split(",") if s.strip()]

# Combine selected and custom skills
all_user_skills = list(user_skills) + custom_skills

# Skill levels
st.sidebar.subheader("Skill Levels")
skill_levels = {}
if all_user_skills:
    st.sidebar.markdown("**Set your proficiency level for each skill:**")
    level_options = {
        "Beginner": 1,
        "Intermediate": 2,
        "Advanced": 3,
        "Expert": 4
    }
    
    for skill in all_user_skills:
        level_label = st.sidebar.selectbox(
            f"{skill}",
            options=list(level_options.keys()),
            index=1,  # Default to Intermediate
            key=f"level_{skill}"
        )
        skill_levels[skill] = level_options[level_label]

if st.sidebar.button("Search Jobs", type="primary", use_container_width=True):
    with st.spinner("Fetching and analyzing jobs..."):
        try:
            data = load_or_fetch_jobs(
                role, location, country,
                date_posted=date_posted,
                work_from_home=work_from_home,
                employment_types=employment_types_str,
                job_requirements=job_requirements_str,
                radius=radius_val
            )
        except Exception as e:
            logger.error(f"Error fetching jobs: {str(e)}")
            st.error(f"Error fetching jobs: {str(e)}")
            st.stop()

        job_results = data.get("data", [])
        
        if not job_results:
            st.warning("No jobs found with the current search parameters.")
            st.stop()

        rows = []
        for job in job_results:
            desc = clean_html(job.get("job_description", ""))
            skills = extract_skills(desc)
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
        
        # === Post-fetch Filtering (Segmentación) ===
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
        
        if len(df) == 0:
            st.warning("No jobs match the selected filters.")
            st.stop()
        
        # Apply clustering
        n_clusters = min(4, len(df))  # Max 4 clusters
        if n_clusters > 1:
            df = cluster_jobs(df, n_clusters=n_clusters)
        else:
            df["cluster"] = 0
        
        df = df.sort_values("match_ratio", ascending=False)
        
        # Prepare data for all tabs
        all_skills_flat = []
        for skills in df["skills_detected"]:
            all_skills_flat.extend(skills if isinstance(skills, list) else [])
        
        # Store in session state for tabs
        st.session_state.df = df
        st.session_state.all_user_skills = all_user_skills
        st.session_state.skill_levels = skill_levels
        st.session_state.missing = missing
        st.session_state.all_skills_flat = all_skills_flat

# Check if we have data to display
if 'df' in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df
    all_user_skills = st.session_state.all_user_skills
    skill_levels = st.session_state.skill_levels
    missing = st.session_state.missing
    all_skills_flat = st.session_state.all_skills_flat
    
    st.sidebar.divider()
    st.sidebar.header("Advanced Analysis")
    enable_graph_analysis = st.sidebar.checkbox("Enable Graph Analysis", value=False)
    show_network_viz = st.sidebar.checkbox("Show Network Visualization", value=True) if enable_graph_analysis else False
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Skills Analysis", "Job Matches", "Recommendations", "Graph Analysis"])
    
    with tab1:
        st.header("Overview Dashboard")
        
        # KPIs with modern styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Jobs", len(df), delta=None)
        with col2:
            avg_match = df["match_ratio"].mean()
            st.metric("Avg Match Ratio", f"{avg_match:.1%}", delta=f"{avg_match*100:.1f}%")
        with col3:
            if "weighted_match_ratio" in df.columns and skill_levels:
                high_match = len(df[df["weighted_match_ratio"] >= 0.5])
            else:
                high_match = len(df[df["match_ratio"] >= 0.5])
            st.metric("High Match Jobs (≥50%)", high_match)
        with col4:
            user_skills_count = len(all_user_skills) if all_user_skills else 0
            avg_level = sum(skill_levels.values()) / len(skill_levels) if skill_levels else 0
            level_label = ["Beginner", "Intermediate", "Advanced", "Expert"][int(avg_level) - 1] if avg_level > 0 else "N/A"
            st.metric("Your Skills", f"{user_skills_count}", delta=level_label)
        
        st.divider()
        
        # Quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Match Ratio Distribution")
            if "weighted_match_ratio" in df.columns and skill_levels:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df["match_ratio"],
                    name="Binary Match",
                    opacity=0.7,
                    nbinsx=20,
                    marker_color="#667eea"
                ))
                fig_hist.add_trace(go.Histogram(
                    x=df["weighted_match_ratio"],
                    name="Weighted Match (with levels)",
                    opacity=0.7,
                    nbinsx=20,
                    marker_color="#764ba2"
                ))
                fig_hist.update_layout(
                    title="Distribution of Match Ratios",
                    xaxis_title="Match Ratio",
                    yaxis_title="Number of Jobs",
                    barmode="overlay",
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18)
                )
            else:
                fig_hist = px.histogram(
                    df,
                    x="match_ratio",
                    nbins=20,
                    labels={"match_ratio": "Match Ratio", "count": "Number of Jobs"},
                    title="Distribution of Match Ratios",
                    color_discrete_sequence=["#b8e994"]
                )
                fig_hist.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18)
                )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Match by Seniority")
            if "seniority" in df.columns:
                seniority_match = df.groupby("seniority")["match_ratio"].mean().reset_index()
                fig_bar = px.bar(
                    seniority_match,
                    x="seniority",
                    y="match_ratio",
                    labels={"seniority": "Seniority Level", "match_ratio": "Avg Match Ratio"},
                    title="Average Match Ratio by Seniority",
                    color="match_ratio",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Skills Analysis")
        
        if all_skills_flat:
            # Most demanded skills
            st.subheader("Most Demanded Skills")
            skill_freq = Counter(all_skills_flat)
            skill_df = pd.DataFrame([
                {
                    "skill": skill,
                    "count": count,
                    "you_have": "Yes" if skill in all_user_skills else "No"
                }
                for skill, count in skill_freq.most_common(15)
            ])
            
            fig = px.bar(
                skill_df,
                x="count",
                y="skill",
                color="you_have",
                orientation="h",
                color_discrete_map={"Yes": "#b8e994", "No": "#e74c3c"},
                labels={"count": "Frequency", "skill": "Skill", "you_have": "You Have"},
                title="Top 15 Most Demanded Skills"
            )
            fig.update_layout(
                height=500, 
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                title_font=dict(color="#b8e994", size=18)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart
            st.subheader("Your Profile vs Ideal Profile")
            if all_user_skills:
                ideal_skills = [skill for skill, _ in skill_freq.most_common(10)]
                comparison_skills = list(set(list(all_user_skills) + ideal_skills))[:10]
                
                if comparison_skills:
                    if skill_levels:
                        user_values = [skill_levels.get(skill, 0) / 4.0 if skill in all_user_skills else 0 
                                      for skill in comparison_skills]
                    else:
                        user_values = [1 if skill in all_user_skills else 0 for skill in comparison_skills]
                    
                    ideal_values = []
                    max_freq = max(skill_freq.values()) if skill_freq else 1
                    for skill in comparison_skills:
                        freq = skill_freq.get(skill, 0)
                        ideal_values.append(freq / max_freq if max_freq > 0 else 0)
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=user_values,
                        theta=comparison_skills,
                        fill='toself',
                        name='Your Profile',
                        line_color='#b8e994',
                        fillcolor='rgba(184, 233, 148, 0.3)'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=ideal_values,
                        theta=comparison_skills,
                        fill='toself',
                        name='Ideal Profile (Market Demand)',
                        line_color='#ffffff',
                        fillcolor='rgba(255, 255, 255, 0.2)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#3a3a4e"),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        showlegend=True,
                        title="Skill Profile Comparison",
                        height=500,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0e0e0"),
                        title_font=dict(color="#b8e994", size=18),
                        legend=dict(font=dict(color="#e0e0e0"))
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            # Missing skills
            st.subheader("Top Missing Skills")
            if missing:
                missing_df = pd.DataFrame(missing).head(10)
                missing_df["priority"] = missing_df["priority"].astype(int)
                
                fig_missing = px.bar(
                    missing_df,
                    x="priority",
                    y="skill",
                    orientation="h",
                    labels={"priority": "Priority (Frequency)", "skill": "Skill"},
                    title="Top 10 Skills You're Missing (Prioritized by Demand)",
                    color="priority",
                    color_continuous_scale="Reds"
                )
                fig_missing.update_layout(
                    height=400, 
                    yaxis={"categoryorder": "total ascending"},
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18)
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("You have all the required skills for these jobs!")
    
    with tab3:
        st.header("Job Matches")
        
        # Format skills_detected for display
        display_df = df.copy()
        display_df["skills_detected"] = display_df["skills_detected"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
        display_df["match_ratio"] = display_df["match_ratio"].apply(lambda x: f"{x:.1%}")
        
        # Select relevant columns for display (excluding apply_link, will add as button)
        cols_to_show = ["title", "company", "city", "seniority", "match_ratio"]
        if "weighted_match_ratio" in display_df.columns:
            cols_to_show.append("weighted_match_ratio")
        cols_to_show.extend(["n_skills_user_has", "n_skills_job", "skills_detected"])
        
        available_cols = [c for c in cols_to_show if c in display_df.columns]
        display_df_display = display_df[available_cols].copy()
        
        # Format weighted_match_ratio for display
        if "weighted_match_ratio" in display_df_display.columns:
            display_df_display["weighted_match_ratio"] = display_df_display["weighted_match_ratio"].apply(lambda x: f"{x:.1%}")
            display_df_display = display_df_display.rename(columns={"weighted_match_ratio": "weighted_match"})
        
        # Add CSS for button styling
        st.markdown("""
        <style>
        .apply-button-link {
            display: inline-block;
            padding: 8px 16px;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 2px solid #b8e994;
            transition: all 0.3s ease;
            font-size: 0.85rem;
        }
        .apply-button-link:hover {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border-color: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(184, 233, 148, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create HTML table with buttons
        html_parts = []
        html_parts.append("""
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; background-color: #2a2a3e; color: #ffffff; margin: 1rem 0;">
        <thead>
            <tr style="background-color: #1a1a2e; color: #b8e994;">
        """)
        
        # Add headers
        headers = list(display_df_display.columns) + ["Apply"]
        for header in headers:
            html_parts.append(f'<th style="padding: 12px; text-align: left; font-weight: 700; border-bottom: 2px solid #3a3a4e;">{header}</th>')
        
        html_parts.append("</tr></thead><tbody>")
        
        # Add rows
        for idx, row in display_df_display.iterrows():
            html_parts.append('<tr style="border-bottom: 1px solid #3a3a4e;">')
            for col in display_df_display.columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                html_parts.append(f'<td style="padding: 12px;">{value}</td>')
            
            # Add Apply button
            apply_link = display_df.loc[idx, "apply_link"] if "apply_link" in display_df.columns else None
            if pd.notna(apply_link) and apply_link:
                button_html = f'<a href="{apply_link}" target="_blank" class="apply-button-link">Apply</a>'
            else:
                button_html = '<span style="color: #a0a0a0;">N/A</span>'
            html_parts.append(f'<td style="padding: 12px;">{button_html}</td>')
            html_parts.append("</tr>")
        
        html_parts.append("</tbody></table></div>")
        
        # Render the table
        st.markdown("".join(html_parts), unsafe_allow_html=True)
        
        # Clustering analysis
        if "cluster" in df.columns and df["cluster"].nunique() > 1:
            st.subheader("Job Clusters")
            cluster_summary = interpret_clusters(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(cluster_summary, hide_index=True)
            with col2:
                cluster_counts = df["cluster"].value_counts().sort_index()
                fig_cluster = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    title="Job Distribution by Cluster",
                    color_discrete_sequence=["#b8e994", "#667eea", "#764ba2", "#fbbf24"]
                )
                fig_cluster.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18),
                    legend=dict(font=dict(color="#e0e0e0"))
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
    
    with tab4:
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
                
                # Calculate percentage of jobs requiring this skill
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

    with tab5:
        if enable_graph_analysis:
            st.header("Skill Network Analysis")
            
            try:
                G = build_skill_cooccurrence_graph(df)
                
                if len(G.nodes()) > 0:
                    centrality_df = compute_centralities(G)
                    
                    st.sidebar.subheader("Community Detection Settings")
                    comm_algorithm = st.sidebar.selectbox(
                        "Algorithm",
                        options=["best", "louvain", "greedy_modularity", "label_propagation"],
                        index=0,
                        help="'best' tries all algorithms and picks the one with highest modularity"
                    )
                    comm_resolution = st.sidebar.slider(
                        "Resolution",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        help="Higher values create more, smaller communities (Louvain only)"
                    )
                    min_comm_size = st.sidebar.slider(
                        "Min Community Size",
                        min_value=1,
                        max_value=5,
                        value=2,
                        help="Merge communities smaller than this size"
                    )
                    
                    communities = detect_communities(G, algorithm=comm_algorithm, 
                                                    resolution=comm_resolution,
                                                    min_community_size=min_comm_size)
                    bridge_skills_list = find_bridge_skills(G, top_n=10)
                    importance_df = get_skill_importance_scores(df, all_user_skills)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Skills", len(G.nodes()))
                    with col2:
                        st.metric("Skill Connections", G.number_of_edges())
                    with col3:
                        st.metric("Communities", len(set(communities.values())) if communities else 0)
                    with col4:
                        avg_importance = importance_df["importance_score"].mean() if not importance_df.empty else 0
                        st.metric("Avg Importance", f"{avg_importance:.2f}")
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top Skills by Importance Score")
                        st.caption("Combines frequency, centrality, and network position")
                        if not importance_df.empty:
                            top_importance = importance_df[["skill", "importance_score", "frequency", "degree", "betweenness"]].head(15)
                            top_importance.columns = ["Skill", "Importance", "Frequency", "Degree", "Betweenness"]
                            top_importance["Importance"] = top_importance["Importance"].apply(lambda x: f"{x:.3f}")
                            top_importance["Degree"] = top_importance["Degree"].apply(lambda x: f"{x:.3f}")
                            top_importance["Betweenness"] = top_importance["Betweenness"].apply(lambda x: f"{x:.3f}")
                            st.dataframe(top_importance, use_container_width=True, hide_index=True)
                        else:
                            st.info("No importance scores available")
                    
                    with col2:
                        st.subheader("Bridge Skills")
                        st.caption("Skills that connect different communities (high betweenness)")
                        if bridge_skills_list:
                            bridge_df = pd.DataFrame({
                                "Skill": bridge_skills_list,
                                "Rank": range(1, len(bridge_skills_list) + 1)
                            })
                            st.dataframe(bridge_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No bridge skills detected")
                    
                    st.divider()
                    
                    st.subheader("Skill Co-occurrence Heatmap")
                    st.caption("Shows which skills frequently appear together (top 20 skills)")
                    if len(G.nodes()) > 0:
                        top_skills_for_heatmap = importance_df["skill"].head(20).tolist() if not importance_df.empty else centrality_df["node"].head(20).tolist()
                        
                        if top_skills_for_heatmap and len(top_skills_for_heatmap) > 1:
                            cooccurrence_matrix = []
                            for skill1 in top_skills_for_heatmap:
                                row = []
                                for skill2 in top_skills_for_heatmap:
                                    if skill1 == skill2:
                                        row.append(0)
                                    elif G.has_edge(skill1, skill2):
                                        weight = G[skill1][skill2].get('weight', 1)
                                        row.append(weight)
                                    else:
                                        row.append(0)
                                cooccurrence_matrix.append(row)
                            
                            fig_cooc = go.Figure(data=go.Heatmap(
                                z=cooccurrence_matrix,
                                x=top_skills_for_heatmap,
                                y=top_skills_for_heatmap,
                                colorscale='Viridis',
                                text=[[f"{val}" if val > 0 else "" for val in row] for row in cooccurrence_matrix],
                                texttemplate="%{text}",
                                textfont={"size": 8},
                                hoverongaps=False,
                                colorbar=dict(title="Co-occurrences", tickfont=dict(color='#e0e0e0'))
                            ))
                            fig_cooc.update_layout(
                                height=600,
                                title="Skill Co-occurrence Matrix",
                                xaxis_title="Skills",
                                yaxis_title="Skills",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e0e0e0"),
                                title_font=dict(color="#b8e994", size=18),
                                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                                yaxis=dict(tickfont=dict(size=9))
                            )
                            st.plotly_chart(fig_cooc, use_container_width=True)
                        else:
                            st.info("Not enough skills for co-occurrence analysis")
                    
                    st.divider()
                    
                    if communities:
                        st.subheader("Skill Communities")
                        community_counts = Counter(communities.values())
                        comm_df = pd.DataFrame([
                            {
                                "Community": f"Community {comm_id}",
                                "Skills Count": count,
                                "Top Skills": ", ".join([skill for skill, cid in communities.items() if cid == comm_id][:8])
                            }
                            for comm_id, count in community_counts.most_common()
                        ])
                        st.dataframe(comm_df, use_container_width=True, hide_index=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Community Sizes")
                            comm_size_data = [
                                {"Community": f"Community {comm_id}", "Size": count}
                                for comm_id, count in sorted(community_counts.items())
                            ]
                            comm_size_df = pd.DataFrame(comm_size_data)
                            fig_comm = px.bar(
                                comm_size_df,
                                x="Community",
                                y="Size",
                                labels={"Size": "Number of Skills", "Community": "Community"},
                                title="Skills per Community",
                                color="Size",
                                color_continuous_scale="Viridis"
                            )
                            fig_comm.update_layout(
                                height=300,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e0e0e0"),
                                title_font=dict(color="#b8e994", size=16),
                                xaxis=dict(tickangle=-45)
                            )
                            st.plotly_chart(fig_comm, use_container_width=True)
                        
                        with col2:
                            st.subheader("Top Skills by Centrality")
                            top_centrality = centrality_df[["node", "degree", "betweenness", "weighted_degree"]].head(10)
                            top_centrality.columns = ["Skill", "Degree", "Betweenness", "Weighted Degree"]
                            top_centrality["Degree"] = top_centrality["Degree"].apply(lambda x: f"{x:.3f}")
                            top_centrality["Betweenness"] = top_centrality["Betweenness"].apply(lambda x: f"{x:.3f}")
                            st.dataframe(top_centrality, use_container_width=True, hide_index=True)
                    
                    if show_network_viz:
                        st.divider()
                        st.subheader("Interactive Network Graph")
                        net = plot_skill_network(G, communities, bridge_skills_list[:10])
                        if net:
                            try:
                                net.save_graph("network.html")
                                with open("network.html", "r", encoding="utf-8") as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=700)
                            except Exception as e:
                                st.error(f"Error displaying network: {e}")
                        else:
                            st.warning("Could not generate network visualization")
                    
                    if all_user_skills:
                        st.divider()
                        st.subheader("Network-Based Skill Recommendations")
                        recommendations = get_skill_recommendations(G, all_user_skills, top_n=10)
                        if not recommendations.empty:
                            recommendations.columns = ["Recommended Skill", "Score", "Reason"]
                            recommendations["Score"] = recommendations["Score"].apply(lambda x: f"{x:.3f}")
                            st.dataframe(recommendations, use_container_width=True, hide_index=True)
                        else:
                            st.info("No recommendations available based on network analysis")
                
                else:
                    st.info("Not enough skills data for network analysis")
                    
            except Exception as e:
                st.error(f"Error in graph analysis: {e}")
                logger.exception("Graph analysis error")
        else:
            st.info("💡 Enable 'Graph Analysis' in the sidebar to see skill network insights and visualizations.")

else:
    # Welcome screen
    st.info("**Get started:** Fill in the search parameters in the sidebar and click 'Search Jobs' to begin your skill gap analysis.")
    
    # Show example
    st.markdown("### How it works:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Search Jobs**
        - Enter your target role
        - Set location and filters
        - Click Search
        """)
    with col2:
        st.markdown("""
        **2. Add Your Skills**
        - Select from the list
        - Add custom skills
        - Set proficiency levels
        """)
    with col3:
        st.markdown("""
        **3. Analyze & Improve**
        - View skill gaps
        - Get recommendations
        - Track your progress
        """)
