import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter


def render_skills_analysis_tab(all_skills_flat, all_user_skills, skill_levels, missing, skill_clusters):
    """
    Render the Skills Analysis tab.
    
    Args:
        all_skills_flat: List of all skills from jobs
        all_user_skills: List of user skills
        skill_levels: Dictionary of skill levels
        missing: List of missing skills
        skill_clusters: Dictionary of skill clusters
    """
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
        
        # Short textual summary of coverage vs market demand
        covered_skills = skill_df[skill_df["you_have"] == "Yes"]["skill"].nunique()
        demanded_skills = skill_df["skill"].nunique()
        missing_skills = demanded_skills - covered_skills
        
        st.markdown(
            f"**You currently cover {covered_skills} / {demanded_skills} of the top market skills** "
            f"shown below. You're missing **{missing_skills} skills** that appear frequently in job offers."
        )
        
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
        
        # Simple radar chart: Your profile vs Market demand (top skills)
        if all_user_skills:
            st.subheader("Your Profile vs Market Demand (Top Skills)")
            top_skills = [skill for skill, _ in skill_freq.most_common(8)]
            max_freq = max(skill_freq.values()) if skill_freq else 1
            
            market_values = [skill_freq[skill] / max_freq for skill in top_skills]
            
            # Map skill levels to numeric values for visualization
            level_to_value = {
                "Basic": 0.25,
                "Intermediate": 0.5,
                "Advanced": 0.75,
                "Expert": 1.0
            }
            
            user_values = []
            for skill in top_skills:
                if skill in all_user_skills:
                    skill_level = skill_levels.get(skill, "Expert")
                    user_values.append(level_to_value.get(skill_level, 1.0))
                else:
                    user_values.append(0.0)
            
            # Close the loop for polar plot
            market_values_loop = market_values + [market_values[0]]
            user_values_loop = user_values + [user_values[0]]
            skills_loop = top_skills + [top_skills[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=user_values_loop,
                theta=skills_loop,
                fill="toself",
                name="You",
                line_color="#b8e994",
                fillcolor="rgba(184, 233, 148, 0.3)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=market_values_loop,
                theta=skills_loop,
                fill="toself",
                name="Market",
                line_color="#ffffff",
                fillcolor="rgba(255, 255, 255, 0.15)",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor="#3a3a4e",
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=True,
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                legend=dict(font=dict(color="#e0e0e0")),
                title=dict(text="")
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Dynamic Skill Clusters
        if skill_clusters:
            st.subheader("Dynamic Skill Clusters")
            st.caption("Skills grouped by semantic similarity using embeddings. Each cluster represents skills that are contextually related.")
            
            # Group skills by cluster
            cluster_groups = {}
            for skill, cluster_id in skill_clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(skill)
            
            # Display clusters
            num_clusters = len(cluster_groups)
            cols = st.columns(min(3, num_clusters))
            
            for idx, (cluster_id, skills) in enumerate(sorted(cluster_groups.items())):
                with cols[idx % len(cols)]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>Cluster {cluster_id}</h4>
                        <p>{', '.join(skills[:8])}{'...' if len(skills) > 8 else ''}</p>
                        <p><em>{len(skills)} skills</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(cluster_groups) > 1:
                cluster_data = []
                for cluster_id, skills in sorted(cluster_groups.items()):
                    for skill in skills:
                        skill_freq = Counter(all_skills_flat)
                        cluster_data.append({
                            "skill": skill,
                            "cluster": f"Cluster {cluster_id}",
                            "frequency": skill_freq.get(skill, 0),
                            "you_have": "Yes" if skill in all_user_skills else "No"
                        })
                
                cluster_df = pd.DataFrame(cluster_data)
                
                # Bar chart showing skills by cluster
                fig_clusters = px.bar(
                    cluster_df,
                    x="cluster",
                    y="frequency",
                    color="you_have",
                    title="Skill Distribution by Cluster",
                    labels={"frequency": "Total Frequency", "cluster": "Cluster"},
                    color_discrete_map={"Yes": "#b8e994", "No": "#e74c3c"}
                )
                fig_clusters.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    title_font=dict(color="#b8e994", size=18)
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Missing skills
        st.subheader("Top Missing Skills")
        if missing:
            missing_df = pd.DataFrame(missing).head(10)
            missing_df["priority"] = missing_df["priority"].astype(int)
            
            st.caption("Click a skill name to search for courses and learning resources.")
            for _, row in missing_df.iterrows():
                skill_name = row["skill"]
                priority = int(row["priority"])
                count = row["count"]
                search_url = f"https://www.google.com/search?q={skill_name.replace(' ', '+')}+course"
                st.markdown(
                    f"- [{skill_name}]({search_url}) — "
                    f"demand in jobs: **{count}**"
                )
            
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

