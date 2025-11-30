import streamlit as st
import pandas as pd
import logging
from collections import Counter

from core.api_client import load_or_fetch_jobs
from core.skills_extraction import clean_html, extract_skills, get_all_skills, get_skills_by_category
from core.analysis import (
    compute_skill_gap, 
    compute_readiness_score, 
    get_recommendations,
    cluster_jobs,
    interpret_clusters
)
from core.seniority_detection import detect_seniority, get_seniority_score
from core.graph_analysis import (
    build_skill_cooccurrence_graph,
    compute_centralities,
    detect_communities,
    get_skill_importance_scores,
    find_bridge_skills
)
from utils.visualization import (
    plot_skill_frequency,
    plot_match_ratio_distribution,
    plot_match_ratio_by_job,
    plot_radar_chart,
    plot_skill_network
)
from utils.export import export_to_csv, export_to_json, export_results_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SkillGap - Advanced Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🧠 SkillGap – Advanced Skill Match & Job Analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("🔍 Job Search Parameters")

role = st.sidebar.text_input("Role", value="data analyst", help="Job role to search for")
location = st.sidebar.text_input("Location", value="Madrid", help="City or location")
country = st.sidebar.text_input("Country code", value="es", help="ISO country code (e.g., 'es', 'us')")

date_posted = st.sidebar.selectbox(
    "Posted date",
    ["all", "today", "3days", "week", "month"],
    help="Filter by posting date"
)

remote = st.sidebar.checkbox("Only remote?", value=False)
work_from_home = True if remote else None

radius = st.sidebar.number_input("Radius (km)", min_value=0.0, value=0.0, step=10.0)
radius_val = radius if radius > 0 else None

st.sidebar.markdown("---")
st.sidebar.header("👤 Your Skills Profile")

# Get skills list
try:
    skills_list = get_all_skills()
    skills_by_category = get_skills_by_category()
except Exception as e:
    st.sidebar.error(f"Error loading skills: {e}")
    skills_list = []
    skills_by_category = {}

# Skills selection with category grouping
if skills_by_category:
    user_skills = []
    for category, skills in skills_by_category.items():
        selected = st.sidebar.multiselect(
            f"{category.title()} Skills",
            options=skills,
            default=[],
            key=f"skills_{category}"
        )
        user_skills.extend(selected)
else:
    user_skills = st.sidebar.multiselect("Select your skills", options=skills_list)

# Advanced options
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Advanced Options")

enable_clustering = st.sidebar.checkbox("Enable Job Clustering", value=True)
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4) if enable_clustering else 4

enable_graph_analysis = st.sidebar.checkbox("Enable Graph Analysis", value=True)
show_network_viz = st.sidebar.checkbox("Show Network Visualization", value=True) if enable_graph_analysis else False

# Search button
if st.sidebar.button("🔎 Search & Analyze", type="primary", use_container_width=True):
    with st.spinner("Fetching jobs and performing analysis..."):
        try:
            # Fetch jobs
            data = load_or_fetch_jobs(
                role, location, country,
                date_posted=date_posted,
                work_from_home=work_from_home,
                radius=radius_val
            )

            job_results = data.get("data", [])

            if not job_results:
                st.warning("No jobs found. Try adjusting your search parameters.")
                st.stop()

            # Process jobs
            rows = []
            all_skills_counter = Counter()
            
            for job in job_results:
                desc = clean_html(job.get("job_description", ""))
                skills = extract_skills(desc)
                
                # Detect seniority
                seniority_info = detect_seniority(
                    job.get("job_title", ""),
                    desc
                )
                
                all_skills_counter.update(skills)
                
                rows.append({
                    "job_id": job.get("job_id"),
                    "title": job.get("job_title"),
                    "company": job.get("employer_name"),
                    "city": job.get("job_city"),
                    "skills_detected": skills,
                    "seniority": seniority_info["level"],
                    "seniority_score": get_seniority_score(
                        job.get("job_title", ""),
                        desc
                    ),
                })

            # Compute skill gap
            rows, missing = compute_skill_gap(rows, user_skills)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Store in session state for filtering
            st.session_state.jobs_df = df
            st.session_state.missing_skills = missing
            st.session_state.user_skills = user_skills
            st.session_state.all_skills_freq = dict(all_skills_counter)

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            logger.exception("Analysis error")
            st.stop()

# Display results if available
if "jobs_df" in st.session_state:
    df = st.session_state.jobs_df
    missing = st.session_state.missing_skills
    user_skills = st.session_state.user_skills
    all_skills_freq = st.session_state.all_skills_freq
    
    # Overview metrics
    st.header("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    readiness = compute_readiness_score(user_skills, missing, len(df))
    
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        st.metric("Readiness Score", f"{readiness['readiness_score']:.1f}%")
    with col3:
        st.metric("Avg Match Ratio", f"{df['match_ratio'].mean():.2%}")
    with col4:
        st.metric("Top Missing Skills", len(missing[:5]))
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Job Matches", 
        "📈 Visualizations", 
        "🕸️ Network Analysis",
        "🎨 Clustering",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("Job Matches")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_match = st.slider("Min Match Ratio", 0.0, 1.0, 0.0, 0.1)
        with col2:
            seniority_filter = st.multiselect(
                "Seniority Level",
                options=df["seniority"].unique(),
                default=[]
            )
        with col3:
            city_filter = st.multiselect(
                "City",
                options=df["city"].dropna().unique(),
                default=[]
            )
        
        # Apply filters
        df_filtered = df[df["match_ratio"] >= min_match]
        if seniority_filter:
            df_filtered = df_filtered[df_filtered["seniority"].isin(seniority_filter)]
        if city_filter:
            df_filtered = df_filtered[df_filtered["city"].isin(city_filter)]
        
        # Display table
        display_cols = ["title", "company", "city", "seniority", "n_skills_job", 
                       "n_skills_user_has", "match_ratio"]
        st.dataframe(
            df_filtered[display_cols].sort_values("match_ratio", ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export to CSV"):
                export_to_csv(df_filtered, "exported_jobs.csv")
                st.success("Exported to exported_jobs.csv")
        with col2:
            if st.button("📥 Export Summary to JSON"):
                export_results_summary(df, missing, user_skills, readiness, "exported_summary.json")
                st.success("Exported to exported_summary.json")
    
    with tab2:
        st.subheader("📈 Visualizations")
        
        # Skill frequency chart
        st.plotly_chart(
            plot_skill_frequency(missing, user_skills, all_skills_freq, top_n=20),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_match_ratio_distribution(df),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_match_ratio_by_job(df, top_n=15),
                use_container_width=True
            )
        
        # Radar chart
        ideal_skills = [{"skill": s["skill"], "count": s["count"]} for s in missing[:20]]
        st.plotly_chart(
            plot_radar_chart(user_skills, ideal_skills),
            use_container_width=True
        )
    
    with tab3:
        if enable_graph_analysis:
            st.subheader("🕸️ Skill Network Analysis")
            
            try:
                # Build graph
                G = build_skill_cooccurrence_graph(df)
                
                if len(G.nodes()) > 0:
                    # Centralities
                    centrality_df = compute_centralities(G)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Skills by Degree Centrality**")
                        st.dataframe(
                            centrality_df[["node", "degree", "betweenness"]].head(10),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.write("**Bridge Skills (High Betweenness)**")
                        bridge_skills = find_bridge_skills(G, top_n=10)
                        st.write(", ".join(bridge_skills))
                    
                    # Communities
                    communities = detect_communities(G)
                    if communities:
                        st.write("**Skill Communities Detected**")
                        community_df = pd.DataFrame([
                            {"skill": skill, "community": comm_id}
                            for skill, comm_id in communities.items()
                        ])
                        st.dataframe(community_df, use_container_width=True)
                    
                    # Network visualization
                    if show_network_viz:
                        st.subheader("Interactive Network Graph")
                        net = plot_skill_network(G, communities, bridge_skills[:10])
                        if net:
                            # Save to HTML and display
                            net.save_graph("network.html")
                            with open("network.html", "r", encoding="utf-8") as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600)
                
                else:
                    st.info("Not enough skills data for network analysis")
                    
            except Exception as e:
                st.error(f"Error in graph analysis: {e}")
                logger.exception("Graph analysis error")
        else:
            st.info("Enable Graph Analysis in sidebar to see network insights")
    
    with tab4:
        if enable_clustering:
            st.subheader("🎨 Job Clustering Analysis")
            
            try:
                # Cluster jobs
                df_clustered = cluster_jobs(df.copy(), n_clusters=n_clusters)
                
                # Interpret clusters
                interpretations = interpret_clusters(df_clustered)
                
                # Display cluster info
                for cluster_id, info in interpretations.items():
                    with st.expander(f"Cluster {cluster_id}: {info['description']} ({info['job_count']} jobs)"):
                        st.write(f"**Top Skills:** {', '.join(info['top_skills'])}")
                        st.write(f"**Average Match Ratio:** {info['avg_match_ratio']:.2%}")
                        
                        # Show jobs in this cluster
                        cluster_jobs_df = df_clustered[df_clustered["cluster"] == cluster_id]
                        st.dataframe(
                            cluster_jobs_df[["title", "company", "match_ratio"]].head(10),
                            use_container_width=True
                        )
                
                # Add cluster to main dataframe
                df["cluster"] = df_clustered["cluster"]
                st.session_state.jobs_df = df
                
            except Exception as e:
                st.error(f"Error in clustering: {e}")
                logger.exception("Clustering error")
        else:
            st.info("Enable Clustering in sidebar to see job clusters")
    
    with tab5:
        st.subheader("💡 Skill Recommendations")
        
        recommendations = get_recommendations(user_skills, missing, top_n=10)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢"
                }
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i}. {rec['skill']}** {priority_color.get(rec['priority'], '⚪')}")
                        st.caption(f"{rec['reason']} | Appears in {rec['job_count']} jobs ({rec['percentage']}%)")
                    st.markdown("---")
        else:
            st.success("🎉 You have all the top required skills!")
        
        # Readiness breakdown
        st.subheader("📊 Readiness Breakdown")
        st.json(readiness)
        
        # Top missing skills table
        st.subheader("Top Missing Skills")
        if missing:
            missing_df = pd.DataFrame(missing[:15])
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("You have all required skills for these jobs!")

else:
    # Initial state
    st.info("👈 Configure your search parameters and skills in the sidebar, then click 'Search & Analyze' to get started!")
    
    # Show skills by category
    try:
        skills_by_category = get_skills_by_category()
        if skills_by_category:
            st.subheader("Available Skills by Category")
            for category, skills in skills_by_category.items():
                with st.expander(f"{category.title()} ({len(skills)} skills)"):
                    st.write(", ".join(skills))
    except:
        pass
