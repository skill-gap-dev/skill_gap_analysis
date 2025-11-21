import requests
import os
from collections import Counter
from textwrap import shorten
from dotenv import load_dotenv
from bs4 import BeautifulSoup # Para limpiar el HTML de la descripción
import spacy
from spacy.matcher import PhraseMatcher

# --- 1. CONFIGURACIÓN Y CARGA DE MODELO ---
load_dotenv()
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")

# Cargamos el modelo de NLP de spaCy (Español pequeño)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    print("Modelo no encontrado. Ejecuta: python -m spacy download es_core_news_sm o instalalo con pip -r requirements.txt")
    exit()

# --- 2. DEFINICIÓN DE LA TAXONOMÍA (HARDCODED PARA PROTOTIPO) ---
# En el futuro, esto vendrá de un CSV (ESCO) o base de datos.
# Normalizamos todo a minúsculas para evitar duplicados al crear patrones.
taxonomy_skills = [
    "Python", "SQL", "R", "Excel", "Tableau", "Power BI", "Java", 
    "Machine Learning", "AWS", "Azure", "Spark", "Hadoop", 
    "Data Visualization", "Estadística", "Inglés", "Git", "Scrum",
    "Communication", "Teamwork", "NoSQL", "Pandas", "Numpy"
]

# Inicializamos el Matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER") # 'LOWER' hace que sea case-insensitive
patterns = [nlp.make_doc(text) for text in taxonomy_skills]
matcher.add("SKILL_LIST", patterns)

# --- 3. FUNCIONES AUXILIARES ---

def clean_html(html_text):
    """Elimina etiquetas HTML de la descripción."""
    if not html_text: return ""
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ")

def extract_skills(description_text):
    """Usa NLP para encontrar skills de la taxonomía en el texto."""
    doc = nlp(description_text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        # Obtenemos el texto original que hizo match
        span = doc[start:end]
        found_skills.add(span.text.title()) # .title() para que quede bonito (ej: python -> Python)
    
    return list(found_skills)

# --- 4. FETCH DE ADZUNA ---

url = "https://api.openwebninja.com/jsearch/search"

headers={
      "x-api-key": os.getenv("API_KEY_JSEARCH")
    }
params={
      "query": "data analysts jobs in Madrid",
      "country": "es",
      "date_posted": "all",
      "page": "1",
      "num_pages": "1"
    }

response = requests.get(url, params=params, headers = headers)
response.raise_for_status()
data = response.json()

# --- 5. PROCESAMIENTO Y RESULTADOS ---
job_results = data.get("data") 

if job_results:
    all_skills_found = []

    # Iteramos sobre la lista 'job_results'
    for job in job_results:
        # 1. Obtener datos básicos
        title = job.get("job_title", "N/A")
        company = job.get("employer_name", "N/A") # 'employer_name' es la clave correcta
        
        # 2. Limpiar descripción
        raw_desc = job.get("job_description", "") # 'job_description' es la clave correcta
        clean_desc = clean_html(raw_desc)
        
        # 3. Extraer Skills
        skills = extract_skills(clean_desc)
        all_skills_found.extend(skills)

        # 4. Imprimir resultado individual
        print("-" * 60)
        print(f"PUESTO: {title} ({company})")
        print(f"Skills detectadas: {', '.join(skills) if skills else 'Ninguna skill de la lista detectada'}")
        # Acortamos el snippet de la descripción para que sea legible en consola
        print(f"Snippet descripción: {shorten(clean_desc, width=150, placeholder='...')}") 

    # --- RESUMEN DEL GAP ---
    print("\n" + "="*60)
    print(" RESUMEN DE DEMANDA (GAP ANALYSIS BASE)")
    print("="*60)
    skill_counts = Counter(all_skills_found)
    num_jobs = len(job_results)
    
    if skill_counts:
        # Ordenamos por las más demandadas
        for skill, count in skill_counts.most_common():
            percentage = (count / num_jobs) * 100
            print(f"{skill:<20} | Presente en {count} ofertas ({percentage:.0f}%)")
    else:
        print("No se encontraron skills. Revisa si tu taxonomía cubre las palabras de las ofertas.")

else:
    print("No jobs found.")