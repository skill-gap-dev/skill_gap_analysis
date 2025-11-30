# 🧠 Skill Gap Analysis - Advanced Edition
**Web Analytics Project — Identifying Skill Gaps for Job Candidates**

## 📌 Overview  
This project identifies the gap between a user's current skill set and the skills required for target jobs. It uses advanced NLP, graph analysis, and machine learning techniques to provide comprehensive insights and recommendations.

**Key Features:**
- 🔍 Intelligent skill extraction with synonym matching
- 📊 Advanced visualizations (radar charts, network graphs, clustering)
- 🕸️ Graph-based analysis (centrality, communities, co-occurrence)
- 🎯 Personalized recommendations with priority scoring
- 📈 Job clustering and segmentation
- 👔 Seniority level detection

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root (see `env.example`):

```env
API_KEY_JSEARCH=your_jsearch_api_key_here
```

Get your API key from [JSearch/OpenWebNinja](https://www.openwebninja.com/).

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows (Git Bash):
source venv/Scripts/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download xx_ent_wiki_sm
```

> **Note for Windows:** If installation hangs, cancel and run: `pip install spacy --only-binary :all:` first.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
skill_gap_analysis/
├── app.py                          # Main Streamlit dashboard
├── core/
│   ├── api_client.py               # JSearch API client with caching
│   ├── skills_extraction.py        # NLP-based skill extraction with synonyms
│   ├── analysis.py                 # Skill gap analysis, clustering, recommendations
│   ├── config.py                   # Configuration constants
│   ├── seniority_detection.py     # Seniority level detection (junior/senior/etc)
│   └── graph_analysis.py           # NetworkX-based graph analysis
├── utils/
│   ├── visualization.py            # Plotly visualization helpers
│   └── export.py                   # Export functions (CSV, JSON)
├── data/
│   ├── taxonomy_skills.csv         # Skill taxonomy with synonyms (50+ skills)
│   └── processed_jobs_*.csv        # Cached processed job data
├── notebooks/
│   └── graph_exploration.ipynb     # Exploratory analysis notebook
├── requirements.txt                # Python dependencies
├── env.example                     # Environment variables template
└── README.md                       # This file
```

## 🎯 Core Functionality

### Skill Extraction
- **NLP-based matching**: Uses spaCy with PhraseMatcher for accurate skill detection
- **Synonym support**: Recognizes multiple variants of the same skill (e.g., "Python", "python3", "py")
- **Multi-language**: Supports English and Spanish job descriptions
- **Taxonomy-driven**: Based on curated taxonomy of 50+ skills with categories

### Skill Gap Analysis
- **Match ratio calculation**: Percentage of required skills the user has
- **Weighted scoring**: Considers skill importance (frequency + centrality)
- **Readiness score**: Overall assessment (0-100%)
- **Missing skills prioritization**: Based on frequency and network importance

### Graph Analysis
- **Co-occurrence networks**: Skills that appear together in job postings
- **Centrality measures**: Degree, betweenness, closeness, eigenvector
- **Community detection**: Louvain algorithm to find skill clusters
- **Bridge skills**: Skills that connect different communities

### Clustering
- **Job clustering**: K-means clustering based on skill requirements
- **Cluster interpretation**: Automatic identification of job types
- **Segmentation**: Filter by location, seniority, remote, etc.

### Visualizations
- **Skill frequency charts**: Bar charts with user skill highlighting
- **Match ratio distribution**: Histogram of job match scores
- **Radar charts**: User profile vs ideal profile comparison
- **Network graphs**: Interactive skill co-occurrence networks
- **Cluster visualizations**: Job type breakdowns

## 🔬 State of the Art & Technical Approach

### Comparison with Existing Solutions

**LinkedIn Skills Assessment:**
- **Similarity**: Both use skill matching and gap analysis
- **Difference**: Our approach uses graph analysis and clustering for deeper insights
- **Advantage**: Open-source, customizable, no API limits for analysis

**Jobscan:**
- **Similarity**: ATS optimization and skill matching
- **Difference**: We focus on skill gap analysis and learning recommendations
- **Advantage**: More analytical depth with graph theory

**Coursera Skill Recommendations:**
- **Similarity**: Skill-based learning paths
- **Difference**: We analyze real job market data, not just course catalogs
- **Advantage**: Market-driven recommendations based on actual job requirements

### Technical Stack

**NLP & Text Processing:**
- **spaCy**: Multi-language NLP model (`xx_ent_wiki_sm`) for text processing
- **PhraseMatcher**: Efficient pattern matching for skill extraction
- **BeautifulSoup**: HTML cleaning from job descriptions

**Graph Analysis:**
- **NetworkX**: Graph construction and analysis algorithms
- **python-louvain**: Community detection (Louvain algorithm)
- **Centrality measures**: Multiple metrics for skill importance

**Machine Learning:**
- **scikit-learn**: K-means clustering, DBSCAN, silhouette analysis
- **Feature engineering**: Binary skill matrices for clustering

**Visualization:**
- **Plotly**: Interactive charts and graphs
- **Pyvis**: Interactive network visualizations
- **Streamlit**: Dashboard framework

### Analytical Techniques

1. **Bipartite Graph Modeling**: Jobs ↔ Skills relationships
2. **Projection to Co-occurrence Graph**: Skill ↔ Skill relationships
3. **Centrality Analysis**: Identifying key skills in the network
4. **Community Detection**: Finding skill clusters (e.g., "Data Science Stack", "Cloud Stack")
5. **Clustering**: Grouping similar job postings
6. **Frequency Analysis**: Market demand for each skill
7. **Weighted Scoring**: Combining multiple metrics for recommendations

### Limitations & Future Work

**Current Limitations:**
- Taxonomy size: 50 skills (expandable to 100+)
- No semantic similarity (only exact/synonym matching)
- No skill level detection (beginner/intermediate/advanced)
- Limited to English/Spanish job descriptions
- No temporal analysis (skill trends over time)

**Future Enhancements:**
- **Embeddings-based matching**: Use sentence transformers for semantic skill matching
- **Skill level detection**: Extract proficiency requirements (e.g., "3+ years Python")
- **Temporal analysis**: Track skill demand trends over time
- **Multi-language expansion**: Support more languages
- **Integration with learning platforms**: Direct links to courses for missing skills
- **Resume parsing**: Automatic skill extraction from user resumes
- **Salary correlation**: Link skills to salary ranges

## 📊 Usage Examples

### Basic Usage

1. **Set your skills** in the sidebar (grouped by category)
2. **Configure search parameters** (role, location, filters)
3. **Click "Search & Analyze"**
4. **Explore results** across 5 tabs:
   - Job Matches: Filtered job listings with match scores
   - Visualizations: Charts and graphs
   - Network Analysis: Skill relationship graphs
   - Clustering: Job type groupings
   - Recommendations: Personalized skill learning suggestions

### Advanced Features

**Graph Analysis:**
- Enable in sidebar → "Enable Graph Analysis"
- View skill networks, communities, and bridge skills
- Interactive network visualization

**Clustering:**
- Enable in sidebar → "Enable Job Clustering"
- Adjust number of clusters (2-8)
- Explore job type interpretations

**Export:**
- Export job matches to CSV
- Export analysis summary to JSON

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests (when implemented)
pytest tests/
```

### Adding New Skills

Edit `data/taxonomy_skills.csv`:
```csv
skill,category,synonyms,language
New Skill,programming,"synonym1|synonym2",en
```

### Extending Analysis

- **New graph metrics**: Add functions to `core/graph_analysis.py`
- **New visualizations**: Add functions to `utils/visualization.py`
- **New clustering methods**: Extend `cluster_jobs()` in `core/analysis.py`

## 📝 API Reference

### Core Functions

**`extract_skills(description: str) -> List[str]`**
Extracts skills from job description using NLP and synonym matching.

**`compute_skill_gap(rows: List[Dict], user_skills: List[str]) -> Tuple[List[Dict], List[Dict]]`**
Computes match ratios and missing skills.

**`build_skill_cooccurrence_graph(jobs_df: pd.DataFrame) -> nx.Graph`**
Builds skill co-occurrence network.

**`cluster_jobs(jobs_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame`**
Clusters jobs by skill requirements.

**`detect_seniority(job_title: str, job_description: str) -> Dict`**
Detects seniority level from job title/description.

## 👥 Authors  
- Carolina López De La Madriz 
- Emma Rodríguez Hervas
- Álvaro Martín Ruiz
- Iker Rosales Saiz

**Web Analytics — 2024/2025**

## 📄 License
See LICENSE file for details.

## 🙏 Acknowledgments
- JSearch/OpenWebNinja for job data API
- spaCy team for NLP models
- NetworkX community for graph analysis tools
