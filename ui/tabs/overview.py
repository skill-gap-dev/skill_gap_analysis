import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_overview_tab(df, skill_levels, all_user_skills):
    """
    Render the Overview dashboard tab.
    
    Args:
        df: DataFrame with job data
        skill_levels: Dictionary of skill levels
        all_user_skills: List of user skills
    """
    st.header("Overview Dashboard")
    
    # 4 columns for rpresenting each kpi
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
        st.metric("Your Skills", f"{user_skills_count}")
    
    # Highlight the single best match as a prominent card
    if not df.empty:
        top_job = df.iloc[0]
        best_match_value = top_job.get(
            "weighted_match_ratio", top_job.get("match_ratio", None)
        )
        if isinstance(best_match_value, (int, float)):
            best_match_pct = f"{best_match_value:.0%}"
        else:
            best_match_pct = "N/A"
        
        apply_link = top_job.get("apply_link")
        apply_html = ""
        if apply_link:
            apply_html = (
                f'<a href="{apply_link}" target="_blank" '
                f'style="display:inline-block;margin-top:0.5rem;'
                f'padding:8px 16px;border-radius:6px;border:2px solid #b8e994;'
                f'background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);'
                f'color:#ffffff;text-decoration:none;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.05em;font-size:0.8rem;">'
                f'View & Apply</a>'
            )
        
        st.markdown(
            f"""
            <div class="recommendation-card">
                <h4>Top Match for You</h4>
                <p><strong>Role:</strong> {top_job.get('title', 'N/A')} at {top_job.get('company', 'N/A')}</p>
                <p><strong>Location:</strong> {top_job.get('city', 'N/A')} &nbsp;|&nbsp; <strong>Match:</strong> {best_match_pct}</p>
                {apply_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
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

