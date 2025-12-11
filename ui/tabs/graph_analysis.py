import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from core.graph_analysis import (
    build_skill_cooccurrence_graph,
    compute_centralities,
    detect_communities,
    find_bridge_skills,
    plot_skill_network,
    get_skill_importance_scores,
)

logger = logging.getLogger(__name__)


def render_graph_analysis_tab(df, all_user_skills, enable_graph_analysis, show_network_viz):
    """
    Render the Graph Analysis tab.
    
    Args:
        df: DataFrame with job data
        all_user_skills: List of user skills
        enable_graph_analysis: Boolean flag to enable graph analysis
        show_network_viz: Boolean flag to show network visualization
    """
    if enable_graph_analysis:
        st.header("Skill Network Analysis")
        st.caption("⭐ indicates skills that you have (in tables, lists, labels and the network graph).")
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
                
                communities = detect_communities(
                    G,
                    algorithm=comm_algorithm,
                    resolution=comm_resolution,
                    min_community_size=min_comm_size
                )
                bridge_skills_list = find_bridge_skills(centrality_df, top_n=10)
                importance_df = get_skill_importance_scores(G, centrality_df, df, all_user_skills)
                
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
                    st.caption("Combines frequency, centrality, and network position. ⭐ marks skills you have.")
                    if not importance_df.empty:
                        top_importance = importance_df[["skill", "importance_score", "frequency", "degree", "betweenness"]].head(15)
                        top_importance.columns = ["Skill", "Importance", "Frequency", "Degree", "Betweenness"]
                        # Highlight user's skills with a star icon
                        top_importance["Skill"] = top_importance["Skill"].apply(
                            lambda s: f"⭐ {s}" if s in all_user_skills else s
                        )
                        top_importance["Importance"] = top_importance["Importance"].apply(lambda x: f"{x:.3f}")
                        top_importance["Degree"] = top_importance["Degree"].apply(lambda x: f"{x:.3f}")
                        top_importance["Betweenness"] = top_importance["Betweenness"].apply(lambda x: f"{x:.3f}")
                        st.dataframe(top_importance, use_container_width=True, hide_index=True)
                    else:
                        st.info("No importance scores available")
                
                with col2:
                    st.subheader("Bridge Skills")
                    st.caption("Skills that connect different communities (high betweenness). ⭐ marks skills you have.")
                    if bridge_skills_list:
                        bridge_df = pd.DataFrame({
                            "Skill": bridge_skills_list,
                            "Rank": range(1, len(bridge_skills_list) + 1)
                        })
                        bridge_df["Skill"] = bridge_df["Skill"].apply(
                            lambda s: f"⭐ {s}" if s in all_user_skills else s
                        )
                        st.dataframe(bridge_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No bridge skills detected")
                
                st.divider()
                
                st.subheader("Skill Co-occurrence Heatmap")
                st.caption("Shows which skills frequently appear together (top 10 skills). ⭐ marks skills you have.")
                if len(G.nodes()) > 0:
                    top_skills_for_heatmap = importance_df["skill"].head(10).tolist() if not importance_df.empty else centrality_df["node"].head(20).tolist()
                    
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
                        
                        x_labels = [
                            f"⭐ {s}" if s in all_user_skills else s
                            for s in top_skills_for_heatmap
                        ]
                        y_labels = x_labels
                        
                        fig_cooc = go.Figure(data=go.Heatmap(
                            z=cooccurrence_matrix,
                            x=x_labels,
                            y=y_labels,
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
                            xaxis=dict(tickangle=-45, tickfont=dict(size=12, color="#e0e0e0")),
                            yaxis=dict(tickfont=dict(size=12, color="#e0e0e0"))
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
                            "Top Skills": ", ".join(
                                [
                                    (f"⭐ {skill}" if skill in all_user_skills else skill)
                                    for skill, cid in communities.items()
                                    if cid == comm_id
                                ][:8]
                            )
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
                        top_centrality["Skill"] = top_centrality["Skill"].apply(
                            lambda s: f"⭐ {s}" if s in all_user_skills else s
                        )
                        top_centrality["Degree"] = top_centrality["Degree"].apply(lambda x: f"{x:.3f}")
                        top_centrality["Betweenness"] = top_centrality["Betweenness"].apply(lambda x: f"{x:.3f}")
                        st.dataframe(top_centrality, use_container_width=True, hide_index=True)
                
                if show_network_viz:
                    st.divider()
                    st.subheader("Interactive Network Graph")
                    st.caption("Showing top 20 skills by importance score. ⭐ marks skills you have.")
                    
                    # Get top 20 skills by importance score
                    if not importance_df.empty:
                        top_20_skills = importance_df["skill"].head(20).tolist()
                    else:
                        # Fallback to centrality if importance_df is empty
                        top_20_skills = centrality_df["node"].head(20).tolist()
                    
                    # Create subgraph with only top 20 skills
                    if top_20_skills and len(top_20_skills) > 0:
                        # Create subgraph containing only top 20 skills and their connections
                        G_subgraph = G.subgraph(top_20_skills).copy()
                        
                        # Filter communities to only include top 20 skills
                        communities_subgraph = {
                            skill: communities.get(skill, 0)
                            for skill in top_20_skills
                            if skill in communities
                        }
                        
                        # Filter bridge skills to only those in top 20
                        bridge_skills_subgraph = [
                            skill for skill in bridge_skills_list[:10]
                            if skill in top_20_skills
                        ]
                        
                        # Filter user skills to only those in top 20
                        user_skills_subgraph = [
                            skill for skill in all_user_skills
                            if skill in top_20_skills
                        ]
                        
                        net = plot_skill_network(
                            G_subgraph,
                            communities_subgraph,
                            highlight_skills=bridge_skills_subgraph,
                            user_skills=user_skills_subgraph,
                        )
                    else:
                        net = plot_skill_network(
                            G,
                            communities,
                            highlight_skills=bridge_skills_list[:10],
                            user_skills=all_user_skills,
                        )
                    if net:
                        try:
                            net.save_graph("data/temp_graph.html")
                            with open("data/temp_graph.html", "r", encoding="utf-8") as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=700)
                        except Exception as e:
                            st.error(f"Error displaying network: {e}")
                    else:
                        st.warning("Could not generate network visualization")
            
            else:
                st.info("Not enough skills data for network analysis")
                
        except Exception as e:
            st.error(f"Error in graph analysis: {e}")
            logger.exception("Graph analysis error")
    else:
        st.info("Enable 'Graph Analysis' in the sidebar to see skill network insights and visualizations.")

