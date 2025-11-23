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
- **Looker Studio (Google Data Studio)** — Dashboard visualization  
- **GitHub** — Project organization & version control

## CLI Prototype (rule_based_matching.py)

We implemented a first CLI prototype (`rule_based_matching.py`) that:
- Asks for role, location and basic filters (remote, employment type, etc.)
- Fetches job postings from JSearch (with simple caching in `data/`)
- Cleans descriptions and performs rule-based skill extraction with spaCy
- Saves a processed CSV for later dashboarding

  
## 👥 Authors  
- Carolina López De La Madriz 
- Emma Rodríguez Hervas
- Álvaro Martín Ruiz
- Iker Rosales Saiz

**Web Analytics — 2024/2025**
