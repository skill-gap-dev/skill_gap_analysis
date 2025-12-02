# 🧠 Skill Gap Analysis  
Web Analytics Project — Identifying Skill Gaps for Job Candidates

## 📌 Overview  
This project aims to identify the gap between a user's current skill set and the skills required for the job they are targeting.  
We fetch job postings from external APIs, currently using **JSearch (OpenWebNinja)** as the main source (the initial prototype was based on **ADZUNA**).    
A dashboard in **Looker Studio (Google Data Studio)** will visually present the missing skills and recommendations to help users prepare and become stronger candidates.

## Configuration

### 🔐 Environment Variables

Create a `.env` file in the project root containing:

```env
# Main job search API (current prototype)
API_KEY_JSEARCH=your_jsearch_api_key

# Legacy Adzuna credentials (optional, kept for future experiments)
APP_ID=your_app_id
APP_KEY=your_app_key
```

Make sure the `.env` file is also listed in your `.gitignore`.

Load the variables in your Python code with:

```python
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY_JSEARCH = os.getenv("API_KEY_JSEARCH")

# Legacy (Adzuna) – not used in the current prototype, but kept for reference
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")
```
### Install the dependencies:
```bash
pip install -r requirements.txt
```
> **Nota para Windows:** Si la instalación se queda bloqueada mucho tiempo, cancela y ejecuta: `pip install spacy --only-binary :all:` antes de reintentar.

### Download the Language Model (Multi-language):
It is necessary to download the NLP model separately to enable analysis in both English and Spanish.
```bash
python -m spacy download xx_ent_wiki_sm
```

## 🎯 Objectives  
- Analyze user-provided skills and job preferences  
- Extract relevant job descriptions using **external job APIs** (currently JSearch / OpenWebNinja)
- Identify skill gaps between user abilities and target job requirements  
- Provide recommendations for upskilling  
- Present results in a clear, interactive dashboard

## 🛠️ Tools & Technologies  
- **JSearch / OpenWebNinja** — Job data extraction  
- **Python** — Data cleaning, structuring, and analysis  
- **Streamlit** — Interactive dashboard  
- **spaCy** — NLP for skill extraction  
- **NetworkX** — Graph analysis and network science  
- **Plotly** — Interactive visualizations  
- **scikit-learn** — Clustering algorithms

## 🚀 Features

### Core Functionality
- **Job Search**: Fetch job postings from JSearch API with caching
- **Skill Extraction**: NLP-based extraction using spaCy with synonym matching
- **Skill Gap Analysis**: Calculate match ratios and identify missing skills
- **Seniority Detection**: Automatic detection of job level (junior/mid/senior)

### Advanced Analytics
- **Graph Analysis**: Skill co-occurrence networks with NetworkX
- **Community Detection**: Identify skill communities using Louvain algorithm
- **Centrality Metrics**: Degree, betweenness, closeness, eigenvector centralities
- **Job Clustering**: K-means clustering to identify job typologies
- **Interactive Visualizations**: Network graphs, radar charts, bar plots

### Dashboard Features
- Interactive filters (role, location, remote, seniority, match ratio)
- Real-time skill gap calculation
- Network visualization of skill relationships
- Cluster analysis of job offers
- Profile comparison (user vs. ideal profile)

## 📊 Project Structure

```
skill_gap_analysis/
├── app.py                    # Streamlit dashboard
├── core/
│   ├── api_client.py        # API client with caching
│   ├── skills_extraction.py # NLP skill extraction
│   ├── analysis.py          # Skill gap & clustering
│   ├── graph_analysis.py    # Network analysis
│   └── config.py            # Configuration
├── data/
│   ├── taxonomy_skills.csv   # Skill taxonomy with synonyms
│   └── processed_jobs_*.csv # Processed job data
├── notebooks/
│   └── graph_exploration.ipynb # Exploration notebook
├── docs/
│   └── STATE_OF_THE_ART.md  # State-of-the-art analysis
└── requirements.txt         # Dependencies
```

## 📚 Documentation

See [docs/STATE_OF_THE_ART.md](docs/STATE_OF_THE_ART.md) for a comprehensive analysis of:
- Existing solutions (LinkedIn, Jobscan, Coursera, etc.)
- Analytical techniques used
- Project limitations
- Future work

  
## 👥 Authors  
- Carolina López De La Madriz 
- Emma Rodríguez Hervas
- Álvaro Martín Ruiz
- Iker Rosales Saiz

**Web Analytics — 2025/2026**
