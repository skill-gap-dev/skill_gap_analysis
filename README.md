# 🧠 Skill Gap Analysis  
Web Analytics Project — Identifying Skill Gaps for Job Candidates

## 📌 Overview  
This project aims to identify the gap between a user's current skill set and the skills required for the job they are targeting.  
Using **ADZUNA** to extract job postings, we aggregate and analyze the required skills for a specific role or team.  
A dashboard in **Looker Studio (Google Data Studio)** will visually present the missing skills and recommendations to help users prepare and become stronger candidates.

## Configuration

### 🔐 Environment Variables

Create a `.env` file in the project root containing:

```
APP_ID=your_app_id
APP_KEY=your_app_key
```

Make sure the `.env` file is also listed in your `.gitignore`.

Load the variables in your Python code with:

```python
from dotenv import load_dotenv
import os

load_dotenv()
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
- Extract relevant job descriptions using **ADZUNA**  
- Identify skill gaps between user abilities and target job requirements  
- Provide recommendations for upskilling  
- Present results in a clear, interactive dashboard

## 🛠️ Tools & Technologies  
- **ADZUNA** — Job data extraction  
- **Python** — Data cleaning, structuring, and analysis  
- **Looker Studio (Google Data Studio)** — Dashboard visualization  
- **GitHub** — Project organization & version control

  
## 👥 Authors  
- Carolina López De La Madriz 
- Emma Rodríguez Hervas
- Álvaro Martín Ruiz
- Iker Rosales Saiz

**Web Analytics — 2024/2025**
