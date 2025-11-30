import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
import streamlit as st

def plot_skill_frequency(missing_skills: List[Dict], user_skills: List[str], 
                        all_skills_freq: Dict[str, int], top_n: int = 15):
    """
    Create a bar chart showing skill frequency, colored by whether user has it.
    """
    # Prepare data
    plot_data = []
    user_skills_set = set(user_skills)
    
    for skill_info in missing_skills[:top_n]:
        skill = skill_info["skill"]
        count = skill_info["count"]
        plot_data.append({
            "skill": skill,
            "count": count,
            "user_has": False
        })
    
    # Add user skills that appear in jobs (for context)
    for skill in user_skills_set:
        if skill in all_skills_freq:
            plot_data.append({
                "skill": skill,
                "count": all_skills_freq[skill],
                "user_has": True
            })
    
    if not plot_data:
        st.info("No skills data to display")
        return None
    
    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.sort_values("count", ascending=True).tail(top_n)
    
    # Create bar chart
    fig = px.bar(
        df_plot,
        x="count",
        y="skill",
        orientation="h",
        color="user_has",
        color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
        labels={"count": "Number of Jobs", "skill": "Skill", "user_has": "You Have It"},
        title="Top Skills in Job Market"
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend_title_text=""
    )
    
    return fig

def plot_match_ratio_distribution(jobs_df: pd.DataFrame):
    """
    Create histogram of match ratios.
    """
    if "match_ratio" not in jobs_df.columns or len(jobs_df) == 0:
        return None
    
    fig = px.histogram(
        jobs_df,
        x="match_ratio",
        nbins=20,
        labels={"match_ratio": "Match Ratio", "count": "Number of Jobs"},
        title="Distribution of Job Match Ratios"
    )
    
    fig.update_layout(
        height=300,
        xaxis_range=[0, 1]
    )
    
    return fig

def plot_match_ratio_by_job(jobs_df: pd.DataFrame, top_n: int = 20):
    """
    Create bar chart of match ratios for top jobs.
    """
    if "match_ratio" not in jobs_df.columns or len(jobs_df) == 0:
        return None
    
    top_jobs = jobs_df.nlargest(top_n, "match_ratio")
    
    fig = px.bar(
        top_jobs,
        x="match_ratio",
        y="title",
        orientation="h",
        labels={"match_ratio": "Match Ratio", "title": "Job Title"},
        title=f"Top {top_n} Job Matches"
    )
    
    fig.update_layout(
        height=min(400, top_n * 25),
        yaxis={"categoryorder": "total ascending"}
    )
    
    return fig

def plot_radar_chart(user_skills: List[str], ideal_skills: List[Dict], 
                    categories: Optional[Dict[str, List[str]]] = None):
    """
    Create radar chart comparing user profile vs ideal profile.
    
    Args:
        user_skills: User's skills
        ideal_skills: List of {skill, count} for ideal profile
        categories: Optional dict mapping category -> list of skills
    """
    if not categories:
        # Default categories
        categories = {
            "Programming": ["Python", "SQL", "R", "Java", "JavaScript", "Pandas", "Numpy"],
            "Analytics": ["Excel", "Power BI", "Tableau", "Data Visualization", "Jupyter"],
            "ML/AI": ["Machine Learning", "Statistics", "Scikit-learn", "TensorFlow", "PyTorch"],
            "Cloud": ["AWS", "Azure", "Docker", "Kubernetes"],
            "Big Data": ["Spark", "Hadoop", "MongoDB", "ETL", "Kafka"],
            "Soft Skills": ["Communication", "Teamwork", "Scrum", "Problem Solving"]
        }
    
    user_skills_set = set(user_skills)
    
    # Calculate scores per category
    categories_list = []
    user_scores = []
    ideal_scores = []
    
    for category, skills in categories.items():
        category_user_skills = [s for s in skills if s in user_skills_set]
        user_score = len(category_user_skills) / len(skills) if len(skills) > 0 else 0
        
        # Ideal score based on frequency
        ideal_skills_dict = {s["skill"]: s.get("count", 0) for s in ideal_skills}
        category_ideal_count = sum(ideal_skills_dict.get(s, 0) for s in skills)
        max_ideal = max([s.get("count", 0) for s in ideal_skills], default=1)
        ideal_score = min(category_ideal_count / max_ideal, 1.0) if max_ideal > 0 else 0
        
        categories_list.append(category)
        user_scores.append(user_score * 100)
        ideal_scores.append(ideal_score * 100)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=user_scores,
        theta=categories_list,
        fill='toself',
        name='Your Profile',
        line_color='#3498db'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=ideal_scores,
        theta=categories_list,
        fill='toself',
        name='Ideal Profile',
        line_color='#e74c3c'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Your Profile vs Ideal Profile",
        height=500
    )
    
    return fig

def plot_skill_network(graph, communities: Optional[Dict[str, int]] = None, 
                      top_skills: Optional[List[str]] = None):
    """
    Create interactive network visualization of skill co-occurrence graph.
    Uses pyvis for interactive HTML output.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pyvis not installed. Install with: pip install pyvis")
        return None
    
    if len(graph.nodes()) == 0:
        return None
    
    # Create pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(graph)
    
    # Color nodes by community if available
    if communities:
        for node in net.nodes:
            node_id = node["id"]
            if node_id in communities:
                comm_id = communities[node_id]
                # Assign color based on community
                colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
                node["color"] = colors[comm_id % len(colors)]
    
    # Highlight top skills
    if top_skills:
        for node in net.nodes:
            if node["id"] in top_skills:
                node["size"] = 30
                node["borderWidth"] = 3
            else:
                node["size"] = 15
    
    # Set edge width based on weight
    for edge in net.edges:
        weight = edge.get("weight", 1)
        edge["width"] = min(weight / 2, 5)
    
    return net

