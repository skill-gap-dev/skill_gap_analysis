"""
Unified CSS styles for the SkillGap application.
"""

def custom_css():
    """
    Returns the complete CSS stylesheet for the application.
    
    Returns:
        str: CSS styles as a string
    """
    return """
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
    
    /* Job table specific styles */
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
    .job-table {
        font-size: 0.9rem;
    }
    .job-table td {
        max-width: 200px;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Footer styling */
    .footer-container {
        margin-top: 4rem;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
        border-top: 2px solid #3a3a4e;
        border-radius: 10px 10px 0 0;
    }
    
    .footer-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1.5rem;
    }
    
    .footer-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .footer-text {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .footer-logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.5rem;
    }
    
    .footer-logo {
        max-height: 40px;
        max-width: 200px;
        object-fit: contain;
        filter: brightness(0.9);
        transition: filter 0.3s ease;
    }
    
    .footer-logo:hover {
        filter: brightness(1.1);
    }

    .footer-logo-link,
    .footer-logo-link:visited,
    .footer-logo-link:hover,
    .footer-logo-link:active {
        text-decoration: none !important;
        border-bottom: none !important;
        color: inherit;
    }



    .footer-logo-text {
        color: #b8e994;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        border-bottom: none;
    }
    
    .footer-divider {
        width: 100%;
        max-width: 300px;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a3a4e, transparent);
    }
    
    .footer-authors {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .footer-names {
        color: #e0e0e0;
        font-size: 0.85rem;
        line-height: 1.6;
        margin: 0;
    }
    
    @media (min-width: 768px) {
        .footer-content {
            flex-direction: row;
            justify-content: space-around;
            align-items: center;
        }
        
        .footer-divider {
            width: 1px;
        }
        
        .footer-names {
            font-size: 0.9rem;
        }
    }

    </style>
    """


def apply_custom_css():
    """
    Applies the custom CSS to the Streamlit app.
    """
    import streamlit as st
    st.markdown(custom_css(), unsafe_allow_html=True)

