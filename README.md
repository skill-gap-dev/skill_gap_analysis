# Skill Gap Analysis
**Authors**: Carolina Lopez de la Madriz, Emma Rodriguez Hervas, Álvaro Martin Ruiz & Iker Rosales Saiz
![SkillGapLogo](assets/skill_gap_logo.jpeg)

A career-analytics platform that identifies the gap between a candidate’s current skill set and the requirements of their target roles.
It retrieves job postings from external APIs (primary: **JSearch / OpenWebNinja**) and provides an interactive **Streamlit dashboard** for skill extraction, gap scoring, graph-based analytics, recommendations, reporting and **real job offers link**!.

![SkillGapWelcomeImage](assets/welcome_screen.jpeg)

## Quickstart

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Download the spaCy multilingual model**

```bash
python -m spacy download xx_ent_wiki_sm
```

3. **Create a `.env` file at the repository root**

```env
# Primary job search API
API_KEY_JSEARCH=your_jsearch_api_key
```

4. **Launch the Streamlit dashboard**

```bash
streamlit run app.py
```

## Key Features

* **Job Search & Caching**

  * Integration with JSearch/OpenWebNinja APIs.
  * Local caching layer to minimize API calls.

* **NLP Skill Extraction**

  * spaCy-based extraction pipeline.
  * Taxonomy-driven synonym normalisation.
  * Custom taxonomy management via `create_taxonomy_file.py`.

* **Skill Gap Analysis**

  * Comparison between user-declared skills and job requirements.
  * Seniority inference (junior / mid / senior).
  * Match ratios and weighted gap scoring.

* **Advanced Analytics**

  * Skill co-occurrence networks (NetworkX).
  * Graph community detection and centrality metrics.
  * Role clustering using scikit-learn.

* **Interactive Dashboard**

  * Streamlit multi-tab interface: overview, skills, matches, recommendations, graph analysis.
  * Real job offers with useful links to apply to the positions that you fit the most!


## Technologies & Libraries

**Language & Frameworks**

* Python 3.10+
* Streamlit (dashboard UI)

**NLP**

* spaCy (`xx_ent_wiki_sm` model)
* Custom taxonomy-based matching

**Data & Analytics**

* pandas, NumPy
* scikit-learn (clustering, vectorisation)
* NetworkX (graph construction & centrality)
* community / python-louvain (community detection)

**Visualisation**

* Streamlit native charts
* Plotly (interactive figures)
* Network visualization exported as HTML

**APIs**

* JSearch / OpenWebNinja (primary job search)


## Project Structure

```
skill_gap_analysis/
├── app.py                     # Streamlit entrypoint orchestrating the UI
├── core/
│   ├── api_client.py          # API integration + caching
│   ├── skills_extraction.py   # NLP extraction + taxonomy logic
│   ├── analysis.py            # Gap scoring and clustering
│   ├── graph_analysis.py      # Network analytics (centrality, detection)
│   └── config.py              # Configuration utilities
├── ui/
│   ├── components.py          # Custom UI components
│   ├── sidebar.py             # Sidebar inputs and preprocessing
│   ├── styles.py              # Dashboard CSS
│   ├── tabs/                  # Individual tab views
│   │   ├── overview.py
│   │   ├── skills_analysis.py
│   │   ├── job_matches.py
│   │   ├── recommendations.py
│   │   └── graph_analysis.py
│   └── welcome.py             # Landing screen
├── data/
│   ├── taxonomy_skills.csv    # Skill taxonomy + synonyms
│   ├── raw_jobs_*.json        # Example API outputs
│   └── temp_graph.html        # Cached graph visualisation
├── create_taxonomy_file.py    # Helper for building/extending the taxonomy
├── requirements.txt
└── LICENSE
```

## How It Works

### 1. Ingestion

* Fetch job postings using JSearch/OpenWebNinja (JSON).
* Local caching prevents redundant external calls.

### 2. NLP Extraction

* spaCy detects raw skills in job descriptions.
* A taxonomy normalises synonyms (e.g., *PyTorch ≈ pytorch*).
* Merged outputs are passed to the analysis pipeline.

### 3. Gap Analysis

* User skill set vs. aggregated job requirements.
* Missing skills ranked by importance and frequency.
* Seniority estimation using job-level text features.

### 4. Advanced Analytics

* Build a co-occurrence graph of skills from job postings.
* Compute degree, betweenness, closeness, eigenvector centralities.
* Detect communities (Louvain) and cluster job roles.

### 5. Visualisation

* Interactive Streamlit tabs show skills, gaps, matches, and graphs.
* Exportable HTML graph visualisation.

---

## Dashboard Preview

### Take a look of the last demo:

[![Demo SkillGap - Intelligent Skill Gap Analysis for Job Seekers - YouTube](https://res.cloudinary.com/marcomontalbano/image/upload/v1765554063/video_to_markdown/images/youtube--xeKAscK2d8o-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=xeKAscK2d8o "Demo SkillGap - Intelligent Skill Gap Analysis for Job Seekers - YouTube")


## Development Notes

* Python 3.10+ recommended for compatibility.
* Keep the `.env` file local — never commit secrets or API keys.
* Large or temporary outputs should be placed under `data/` (gitignored except taxonomy samples).
* Codebase follows a modular architecture: **core logic** separated from **UI components**.

## Limitations

- **Optimised for Data & AI roles:** The current pipeline and taxonomy ensure reliable performance mainly for technical positions in Data & AI. Other domains may yield incomplete or noisier results.

- **Taxonomy-dependent adaptability:** Expanding the taxonomy is required to support additional industries. With a well-designed taxonomy, the system can be adapted to virtually any job family.

- **Model and API variability:** Skill extraction depends on a general-purpose spaCy model and on the structure of external job APIs, which may introduce inconsistencies or reduce precision in certain postings.

## Authors

* Carolina López De La Madriz
* Emma Rodríguez Hervas
* Álvaro Martín Ruiz
* Iker Rosales Saiz
