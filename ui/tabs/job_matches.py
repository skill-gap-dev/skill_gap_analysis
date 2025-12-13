import streamlit as st
import pandas as pd


def format_skills_list(skills, max_length=150):
    """Format skills list, truncating if too long"""
    if isinstance(skills, list):
        skills_str = ", ".join(skills)
    else:
        skills_str = str(skills)
    
    if len(skills_str) > max_length:
        truncated = skills_str[:max_length].rsplit(',', 1)[0] + "..."
        return truncated
    return skills_str


def render_job_matches_tab(df):
    """
    Render the Job Matches tab.
    
    Args:
        df: DataFrame with job data
    """
    st.header("Job Matches")
    
    # Initialize pagination state
    if 'job_matches_page' not in st.session_state:
        st.session_state.job_matches_page = 1
    
    # Format skills_detected for display (truncate if too long for better performance with 100+ jobs)
    display_df = df.copy()
    
    display_df["skills_detected"] = display_df["skills_detected"].apply(format_skills_list)
    display_df["match_ratio"] = display_df["match_ratio"].apply(lambda x: f"{x:.1%}")
    
    # Select columns to display
    cols_to_show = ["title", "company", "city", "seniority", "match_ratio"]
    if "weighted_match_ratio" in display_df.columns:
        cols_to_show.append("weighted_match_ratio")
    cols_to_show.extend(["n_skills_user_has", "n_skills_job", "skills_detected"])
    
    available_cols = [c for c in cols_to_show if c in display_df.columns]
    display_df_display = display_df[available_cols].copy()
    
    # Format weighted_match_ratio for display
    if "weighted_match_ratio" in display_df_display.columns:
        display_df_display["weighted_match_ratio"] = display_df_display["weighted_match_ratio"].apply(lambda x: f"{x:.1%}")
    
    # Column name mapping to English (more descriptive)
    column_mapping = {
        "title": "Job Title",
        "company": "Company",
        "city": "City",
        "seniority": "Seniority",
        "match_ratio": "Match Ratio",
        "weighted_match_ratio": "Weighted Match Ratio",
        "n_skills_user_has": "Skills You Have",
        "n_skills_job": "Skills Required",
        "skills_detected": "Skills Detected"
    }
    
    # Rename columns
    display_df_display = display_df_display.rename(columns=column_mapping)
    
    # Pagination settings
    rows_per_page = 10
    total_rows = len(display_df_display)
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    
    # Display pagination info and controls
    col_info, col_page = st.columns([2, 2])
    
    with col_info:
        st.caption(f"Showing {total_rows} jobs in total")
    
    with col_page:
        page_input = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.job_matches_page,
            key="page_input_job_matches",
            label_visibility="collapsed"
        )
        if page_input != st.session_state.job_matches_page:
            st.session_state.job_matches_page = page_input
            st.rerun()
        st.caption(f"Page {st.session_state.job_matches_page} of {total_pages}")
    
    # Calculate pagination slice
    start_idx = (st.session_state.job_matches_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    paginated_df = display_df_display.iloc[start_idx:end_idx]
    paginated_original_df = display_df.iloc[start_idx:end_idx]
    
    # Create HTML table with buttons
    html_parts = []
    html_parts.append("""
    <div style="overflow-x: auto;">
    <table class="job-table" style="width: 100%; border-collapse: collapse; background-color: #2a2a3e; color: #ffffff; margin: 1rem 0;">
    <thead>
        <tr style="background-color: #1a1a2e; color: #b8e994;">
    """)
    
    # Add headers
    headers = list(paginated_df.columns) + ["Apply"]
    for header in headers:
        html_parts.append(f'<th style="padding: 12px; text-align: left; font-weight: 700; border-bottom: 2px solid #3a3a4e;">{header}</th>')
    
    html_parts.append("</tr></thead><tbody>")
    
    # Add rows
    for i, (display_idx, row) in enumerate(paginated_df.iterrows()):
        original_idx = paginated_original_df.index[i]
        html_parts.append('<tr style="border-bottom: 1px solid #3a3a4e;">')
        for col in paginated_df.columns:
            value = str(row[col]) if pd.notna(row[col]) else ""
            # Escape HTML to prevent XSS
            value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<td style="padding: 12px;">{value}</td>')
        
        # Add Apply button - get from original display_df using the original index
        apply_link = display_df.loc[original_idx, "apply_link"] if "apply_link" in display_df.columns else None
        if pd.notna(apply_link) and apply_link:
            button_html = f'<a href="{apply_link}" target="_blank" class="apply-button-link">Apply</a>'
        else:
            button_html = '<span style="color: #a0a0a0;">N/A</span>'
        html_parts.append(f'<td style="padding: 12px;">{button_html}</td>')
        html_parts.append("</tr>")
    
    html_parts.append("</tbody></table></div>")
    
    # Render the table
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    # Show pagination info at bottom
    if total_pages > 1:
        st.caption(f"Showing jobs {start_idx + 1} to {end_idx} of {total_rows}")

