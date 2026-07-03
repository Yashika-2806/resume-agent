import json
import math
import os
import re
import requests
from pydantic import BaseModel, Field
from pypdf import PdfReader
import gdown
from openai import OpenAI, AzureOpenAI
from groq import Groq
from anthropic import Anthropic


class ResumeData(BaseModel):
    # ── S_hygiene ──
    total_page_count: int = Field(default=1)
    extracted_links_array: list[str] = Field(default_factory=list)   # all href/urls found
    raw_email_string: str = Field(default="")
    detected_section_headers: list[str] = Field(default_factory=list)

    # ── S_realization ──
    skills_section_keywords: list[str] = Field(default_factory=list)
    project_descriptions_text_corpus: str = Field(default="")
    experience_descriptions_text_corpus: str = Field(default="")

    # ── S_complexity ──
    project_titles: list[str] = Field(default_factory=list)
    project_tech_keywords: list[list[str]] = Field(default_factory=list)  # per-project
    architectural_regex_flags: list[bool] = Field(default_factory=list)   # per-project

    # ── S_impact ──
    total_bullet_points_count: int = Field(default=0)
    metric_regex_match_count: int = Field(default=0)
    regex_extracted_numeric_values: list[int] = Field(default_factory=list)  # V_b values

    # ── S_production ──
    project_count: int = Field(default=0)
    code_repository_urls: list[str] = Field(default_factory=list)
    deployment_live_urls: list[str] = Field(default_factory=list)

    # ── S_clarity ──
    buzzword_frequency_map: dict[str, int] = Field(default_factory=dict)

    # ── S_domain ──
    domain_classification_vector: list[str] = Field(default_factory=list)

    # ── S_velocity ──
    experience_timeline_intervals: list[dict] = Field(default_factory=list)  # [{role, months, type}]

    # ── meta ──
    candidate_name: str = Field(default="Unknown")
    btech_year: int = Field(default=3)   # 2, 3, or 4


#  SYSTEM PROMPT

SYSTEM_PROMPT = """You are a precise resume data extractor for B.Tech student resumes.
Return ONLY a valid JSON object — no markdown, no explanation, no extra keys.

Extract EXACTLY these fields:

total_page_count (int): Number of pages in the resume.

extracted_links_array (array of strings): Every URL/link found (GitHub, LinkedIn, portfolio, Vercel, Netlify, etc.).

raw_email_string (string): The email address found on the resume.

detected_section_headers (array of strings): All section headings found, e.g. ["Education","Projects","Skills","Experience"].

skills_section_keywords (array of strings): ONLY skills listed in the dedicated Skills section.

project_descriptions_text_corpus (string): All text from the Projects section concatenated.

experience_descriptions_text_corpus (string): All text from Experience/Internships section concatenated.

project_titles (array of strings): Title of each project listed.

project_tech_keywords (array of arrays of strings): For each project, the tech keywords used IN THAT PROJECT (same order as project_titles).

architectural_regex_flags (array of booleans): For each project, true if it uses any of: WebSockets, Kafka, Docker, Kubernetes, Redis, CI/CD, gRPC, Microservices, Distributed Systems, AWS, GCP, Azure, Celery, RabbitMQ. Same order as project_titles.

total_bullet_points_count (int): Total bullet points across Projects and Experience sections.

metric_regex_match_count (int): Count of bullet points containing numbers/percentages/ms/users/$. e.g. "40%", "500 users", "200ms".

regex_extracted_numeric_values (array of integers): For each metric bullet, extract the numeric value:
  - Percentages: raw integer (40% -> 40)
  - Users/scale: raw count (500 users -> 500)
  - Latency: ms value (200ms -> 200)
  - 1st place -> 100, Top 10% -> 50

project_count (int): Total number of projects.

code_repository_urls (array of strings): Only GitHub/GitLab repo links for projects.

deployment_live_urls (array of strings): Only live deployment links (Vercel, Netlify, Heroku, AWS link, custom domain for a project).

buzzword_frequency_map (object): Count occurrences of these EXACT words anywhere in the resume:
  ["passionate","detail-oriented","synergy","motivated","hardworking","team player","go-getter","self-starter","results-driven","dynamic","innovative","proactive"]
  Only include words that appear at least once. e.g. {"passionate": 2, "motivated": 1}

domain_classification_vector (array of strings): Map skills_section_keywords to these domains only:
  ["Web Development","AI/ML","DevOps","Web3","CyberSecurity","Mobile","Systems","Data Engineering","UI/UX"]
  List only UNIQUE domains found. e.g. ["Web Development","DevOps"]

experience_timeline_intervals (array of objects): Each experience entry as:
  {"role": "SDE Intern at Google", "months": 3, "type": "internship"}
  type must be one of: "internship", "freelance", "tech_lead", "member"
  Classify campus technical roles as "tech_lead", non-technical club roles as "member".

candidate_name (string): Full name of the candidate.

btech_year (int): 2, 3, or 4. Infer from graduation year or year of study mentioned. Default 3.
"""

#  LLM EXTRACTION

def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def _get_llm_client_and_model(
    api_key: str,
    provider: str,
    model: str = None,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview",
    base_url: str = None
):
    if provider is not None:
        prov = provider.lower()
        if "claude" in prov or "anthropic" in prov:
            provider = "claude"
        elif "azure" in prov:
            provider = "azure"
        elif "proxy" in prov:
            provider = "custom_proxy"
        elif "groq" in prov:
            provider = "groq"
        else:
            provider = "openai"
    else:
        if api_key.strip().startswith("gsk_"):
            provider = "groq"
        elif api_key.strip().startswith("sk-ant-"):
            provider = "claude"
        else:
            provider = "openai"

    if provider == "openai":
        client = OpenAI(api_key=api_key)
        if model is None:
            model = "gpt-4o-mini"
    elif provider == "custom_proxy":
        client = OpenAI(api_key=api_key, base_url=base_url)
        if model is None:
            model = "gpt-4o-mini"
    elif provider == "azure":
        client = AzureOpenAI(
            api_key=api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint
        )
        if model is None:
            model = azure_deployment
    elif provider == "claude":
        client = Anthropic(api_key=api_key)
        if model is None:
            model = "claude-3-5-sonnet-20241022"
    else:
        client = Groq(api_key=api_key)
        if model is None:
            model = "llama-3.3-70b-versatile"

    return client, provider, model

def extract_resume_data(
    resume_text: str,
    api_key: str,
    btech_year: int = 3,
    provider: str = None,
    model: str = None,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview",
    base_url: str = None
) -> ResumeData:
    client, provider, model = _get_llm_client_and_model(
        api_key=api_key,
        provider=provider,
        model=model,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version,
        base_url=base_url
    )

    if provider == "claude":
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Extract data from this resume:\n\n{resume_text[:7000]}"}
            ]
        )
        raw = response.content[0].text
        raw = clean_json_string(raw)
    else:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract data from this resume:\n\n{resume_text[:7000]}"}
            ]
        )
        raw = response.choices[0].message.content
        
    parsed = json.loads(raw)
    
    # Inject the Orchestrator's calculated btech_year into the LLM's JSON output
    # so the ResumeData Pydantic model can pass it down to compute_score()
    parsed["btech_year"] = btech_year
    
    return ResumeData(**parsed)

def _get_subscore(score_dict: dict, key: str) -> float:
    if "scores" in score_dict and isinstance(score_dict["scores"], dict):
        return score_dict["scores"].get(key, 0.0)
    if "sub_scores" in score_dict and isinstance(score_dict["sub_scores"], dict):
        short_key = key.replace("S_", "").lower()
        if short_key in score_dict["sub_scores"]:
            return score_dict["sub_scores"][short_key]
        for k, v in score_dict["sub_scores"].items():
            if k.lower() == short_key or k.lower() == key.lower():
                return v
    if key in score_dict:
        return score_dict[key]
    short_key = key.replace("S_", "")
    if short_key in score_dict:
        return score_dict[short_key]
    return 0.0

def generate_score_explanation(
    resume_text: str,
    score_dict: dict,
    api_key: str,
    provider: str = None,
    model: str = None,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview",
    base_url: str = None
) -> str:
    """
    Queries the LLM to generate a qualitative review explaining the score,
    strengths, weaknesses, and concrete recommendations.
    """
    client, provider, model = _get_llm_client_and_model(
        api_key=api_key,
        provider=provider,
        model=model,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version,
        base_url=base_url
    )

    scores_summary = f"""
Calculated Scores:
- Overall Score: {score_dict.get('final_score', 0):.2f}/100
- Structural Hygiene: {_get_subscore(score_dict, 'S_hygiene'):.2f}/100
- Tech Realization: {_get_subscore(score_dict, 'S_realization'):.2f}/100
- Project Complexity: {_get_subscore(score_dict, 'S_complexity'):.2f}/100
- Metric Impact: {_get_subscore(score_dict, 'S_impact'):.2f}/100
- Production Readiness: {_get_subscore(score_dict, 'S_production'):.2f}/100
- Linguistic Clarity: {_get_subscore(score_dict, 'S_clarity'):.2f}/100
- Domain Focus: {_get_subscore(score_dict, 'S_domain'):.2f}/100
- Activity Velocity: {_get_subscore(score_dict, 'S_velocity'):.2f}/100
"""

    prompt = f"""You are an expert resume reviewer and career coach.
Analyze the following resume and its calculated multi-dimensional scores, and generate a brief, professional explanation:
1. Explain why the score is high/low in specific dimensions.
2. Outline what is missing or what could be improved.
3. Write in a clear, constructive tone (bullet points).

{scores_summary}

Resume Text:
{resume_text[:5000]}
"""

    if provider == "claude":
        response = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
NOISE_WORDS = {
    "passionate", "detail-oriented", "synergy", "motivated", "hardworking",
    "team player", "go-getter", "self-starter", "results-driven", "dynamic",
    "innovative", "proactive"
}

TIER3_SKILLS = {"golang","go","docker","kubernetes","redis","kafka","grpc","aws","gcp","azure",
                "tensorflow","pytorch","spark","hadoop","elasticsearch","rabbitmq","celery",
                "websockets","microservices","ci/cd","jenkins","terraform"}
TIER2_SKILLS = {"python","java","javascript","typescript","react","nodejs","node.js","sql",
                "mongodb","postgresql","mysql","git","spring","fastapi","flask","django",
                "express","graphql","rest","linux","bash","c#","kotlin","swift"}
TIER1_SKILLS = {"html","css","markdown","bootstrap","figma","canva","xml","json","jquery"}

ARCH_KEYWORDS = {"websockets","kafka","docker","kubernetes","redis","ci/cd","grpc",
                 "microservices","distributed","aws","gcp","azure","celery","rabbitmq"}

ROLE_WEIGHTS = {"internship": 15, "freelance": 10, "tech_lead": 10, "member": 3}

WEIGHTS = {
    2: {"hyg": 0.25, "real": 0.25, "comp": 0.20, "imp": 0.05, "prod": 0.10, "clar": 0.05, "dom": 0.05, "vel": 0.05},
    3: {"hyg": 0.15, "real": 0.20, "comp": 0.25, "imp": 0.10, "prod": 0.15, "clar": 0.05, "dom": 0.05, "vel": 0.05},
    4: {"hyg": 0.05, "real": 0.10, "comp": 0.30, "imp": 0.20, "prod": 0.15, "clar": 0.05, "dom": 0.05, "vel": 0.10},
}


def _skill_difficulty(skill: str) -> int:
    s = skill.lower().strip()
    if s in TIER3_SKILLS: return 10
    if s in TIER2_SKILLS: return 5
    return 2  


def _project_tier(tech_keywords: list[str], arch_flag: bool) -> int:
    """Classify a project into Tier 1/2/3 complexity score."""
    if arch_flag:
        return 100 
    kw = {k.lower() for k in tech_keywords}
    
    if kw & TIER3_SKILLS:
        return 100
    
    has_backend = bool(kw & {"nodejs","node.js","express","django","flask","fastapi","spring","java","python","golang"})
    has_db = bool(kw & {"mongodb","postgresql","mysql","sql","redis","firebase","supabase"})
    if has_backend and has_db:
        return 65
    return 25 




def compute_score(data: ResumeData, config_overrides: dict = None) -> dict:
    """
    Evaluates the Pydantic ResumeData object using mathematical constraints.
    Injects orchestrator parameters or falls back to global defaults.
    """
    # ── 0. Safe Configuration Initialization ──
    config = config_overrides or {}
    
    # Isolate Configuration Blocks
    consts = config.get("constants", {})
    pens = config.get("penalties", {})
    roles = config.get("role_weights", {})
    
    # Dynamic weight map fallback
    weights_map = config.get("weights", {
        2: {"hyg": 0.25, "real": 0.25, "comp": 0.20, "imp": 0.05, "prod": 0.10, "clar": 0.05, "dom": 0.05, "vel": 0.05},
        3: {"hyg": 0.15, "real": 0.20, "comp": 0.25, "imp": 0.10, "prod": 0.15, "clar": 0.05, "dom": 0.05, "vel": 0.05},
        4: {"hyg": 0.05, "real": 0.10, "comp": 0.30, "imp": 0.20, "prod": 0.15, "clar": 0.05, "dom": 0.05, "vel": 0.10}
    })

    # Safely Extract Constants for Formulas
    alpha = consts.get("alpha", 5.0)
    beta = consts.get("beta", 12.0)
    omega = consts.get("omega", 15.0)
    eps = consts.get("eps", 1.0)
    
    # Safely Extract Hygiene Penalties
    pen_page = pens.get("hygiene_page_pen", 50)
    pen_link = pens.get("hygiene_link_pen", 15)
    pen_email = pens.get("hygiene_email_pen", 25)
    pen_sec = pens.get("hygiene_sec_pen", 20)

    # ── 1. S_hygiene ──
    P = max(data.total_page_count, 1)
    links_lower = [l.lower() for l in data.extracted_links_array]
    has_github = any("github" in l for l in links_lower)
    has_linkedin = any("linkedin" in l for l in links_lower)
    L_missing = (0 if has_github else 1) + (0 if has_linkedin else 1)
    
    email = data.raw_email_string.lower()
    E_generic = 1 if any(c.isdigit() for c in email.split("@")[0]) or \
                     any(w in email for w in ["cool","coder","gamer","noob","pro","god","king","boss"]) else 0
                     
    mandatory = {"education", "projects", "skills"}
    found = {h.lower() for h in data.detected_section_headers}
    X_missing = len(mandatory - found)
    
    S_hygiene = max(0, 100 - pen_page * max(0, P - 1) - pen_link * L_missing - pen_email * E_generic - pen_sec * X_missing)

    # ── 2. S_realization ──
    declared = set(k.lower().strip() for k in data.skills_section_keywords)
    corpus = (data.project_descriptions_text_corpus + " " + data.experience_descriptions_text_corpus).lower()
    applied = {k for k in declared if k in corpus}
    intersect = declared & applied

    sum_intersect = sum(math.log(_skill_difficulty(k) + 1) for k in intersect)
    sum_declared = sum(math.log(_skill_difficulty(k) + 1) for k in declared) + eps
    S_realization = (sum_intersect / sum_declared) * 100

    # ── 3. S_complexity ──
    if data.project_titles:
        tiers = []
        for i, title in enumerate(data.project_titles):
            tech = data.project_tech_keywords[i] if i < len(data.project_tech_keywords) else []
            arch = data.architectural_regex_flags[i] if i < len(data.architectural_regex_flags) else False
            
            # Assuming _project_tier helper exists in scope
            tiers.append(_project_tier(tech, arch))
            
        max_cj = max(tiers)
        J = len(data.project_titles)
        S_complexity = min(100, max_cj + alpha * math.log(J + 1))
    else:
        S_complexity = 0.0

    # ── 4. S_impact ──
    values = data.regex_extracted_numeric_values or []
    S_impact = min(100, beta * sum(math.log10(v + 1) for v in values if v > 0))

    # ── 5. S_production ──
    J_total = max(data.project_count, 1)
    J_code = len(data.code_repository_urls)
    J_deploy = len(data.deployment_live_urls)
    S_production = ((J_code + J_deploy) / (2 * J_total)) * 100

    # ── 6. S_clarity ──
    bmap = data.buzzword_frequency_map or {}
    deduction = omega * sum(math.log(count + 1) for count in bmap.values() if count > 0)
    S_clarity = max(0, 100 - deduction)

    # ── 7. S_domain ──
    unique_domains = len(set(data.domain_classification_vector))
    total_skills = len(data.skills_section_keywords) + eps
    S_domain = 100 * (1 - unique_domains / total_skills)
    S_domain = max(0, min(100, S_domain))

    # ── 8. S_velocity ──
    velocity_sum = sum(
        # Fall back to 3 if the role type isn't found in our config
        e.get("months", 0) * roles.get(e.get("type", "member"), 3)
        for e in data.experience_timeline_intervals
    )
    S_velocity = min(100, velocity_sum)

    # ── FINAL SCORE ──
    year = data.btech_year if data.btech_year in weights_map else 3
    
    # Safe fallback to year 3 matrix if the year key is somehow missing
    W = weights_map.get(year, weights_map.get(3)) 

    S_final = (
        W.get("hyg", 0.15)  * S_hygiene +
        W.get("real", 0.20) * S_realization +
        W.get("comp", 0.25) * S_complexity +
        W.get("imp", 0.10)  * S_impact +
        W.get("prod", 0.15) * S_production +
        W.get("clar", 0.05) * S_clarity +
        W.get("dom", 0.05)  * S_domain +
        W.get("vel", 0.05)  * S_velocity
    )
    S_final = round(min(100, max(0, S_final)), 2)

    return {
        "final_score": S_final,
        "btech_year": year,
        "weights": W,
        "S_hygiene": round(S_hygiene, 2),
        "S_realization": round(S_realization, 2),
        "S_complexity": round(S_complexity, 2),
        "S_impact": round(S_impact, 2),
        "S_production": round(S_production, 2),
        "S_clarity": round(S_clarity, 2),
        "S_domain": round(S_domain, 2),
        "S_velocity": round(S_velocity, 2),
        # debug helpers
        "L_missing": L_missing,
        "E_generic": E_generic,
        "X_missing": X_missing,
        "buzzwords_found": bmap,
    }


# import os
# from pypdf import PdfReader  # Ensuring your PDF Reader import is present

def execute_resume_agent(pdf_path: str, btech_year: int, config_overrides: dict = None) -> dict:
    """
    Thread-safe entry point for the Orchestrator.
    Handles file I/O, API key injection, and math engine execution.
    """
    try:
        # Check if path was passed and actually exists
        if not pdf_path or not os.path.exists(pdf_path):
            return {
                "agent": "resume",
                "status": "failed",
                "error_log": f"PDF file not found or path was None: {pdf_path}",
                "final_score": 0
            }

        # 1. Extract Text from PDF
        reader = PdfReader(pdf_path)
        text_corpus = " ".join(page.extract_text() for page in reader.pages if page.extract_text())

        # 2. Secure API key
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")

        # 3. Execute LLM Extraction
        resume_data = extract_resume_data(text_corpus, api_key, btech_year)

        # 4. Execute the fully updated Math Engine
        score_data = compute_score(resume_data, config_overrides=config_overrides)

        return {
            "agent": "resume",
            "status": "success",
            "final_score": score_data["final_score"],
            "sub_scores": {
                "S_hygiene": score_data["S_hygiene"],
                "S_realization": score_data["S_realization"],
                "S_complexity": score_data["S_complexity"],
                "S_impact": score_data["S_impact"],
                "S_production": score_data["S_production"],
                "S_clarity": score_data["S_clarity"],
                "S_domain": score_data["S_domain"],
                "S_velocity": score_data["S_velocity"]
            },
            "narrative_context": {
                "buzzwords_found": score_data.get("buzzwords_found", {}),
                "generic_email_flag": score_data.get("E_generic", 0),
                "missing_sections": score_data.get("X_missing", 0)
            }
        }
    except Exception as e:
        return {
            "agent": "resume",
            "status": "failed",
            "error_log": f"Resume Thread Error: {str(e)}",
            "final_score": 0
        }

def download_single_gdrive_file(url: str, dest_path: str) -> bool:
    """
    Downloads a single Google Drive file using requests.
    Handles larger files and confirmation tokens safely.
    Returns True if successful, False otherwise.
    """
    file_id = None
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
    else:
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
        if match:
            file_id = match.group(1)
            
    if not file_id:
        return False
        
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    try:
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        if not token:
            match_token = re.search(r'confirm=([a-zA-Z0-9_]+)', response.text)
            if match_token:
                token = match_token.group(1)
                
        if token:
            response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
            
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                if "Access Denied" in response.text or "login" in response.text or "sign in" in response.text.lower():
                    return False
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
            return os.path.exists(dest_path) and os.path.getsize(dest_path) > 100
    except Exception:
        pass
    return False

def download_gdrive_file_safe(url: str, dest_path: str) -> bool:
    """
    Downloads a single file from Google Drive using the requests API first,
    falling back to gdown if it fails.
    """
    # 1. Try our custom requests downloader
    success = download_single_gdrive_file(url, dest_path)
    if success:
        return True
    # 2. Fall back to gdown
    try:
        filepath = gdown.download(url, quiet=True, fuzzy=True, output=dest_path)
        if filepath and os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            return True
    except Exception:
        pass
    return False

def download_from_gdrive(url: str, dest_dir: str) -> list[str]:
    """
    Downloads files/folders from Google Drive.
    Returns list of downloaded file paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = url.strip()
    
    # Check if folder or file
    is_folder = "drive.google.com/drive/folders" in url or ("drive.google.com/drive/u/" in url and "/folders/" in url)
    
    downloaded_paths = []
    try:
        if is_folder:
            files = gdown.download_folder(url, output=dest_dir, quiet=True, remaining_ok=True, use_cookies=False)
            if files:
                for f in files:
                    if os.path.exists(f):
                        downloaded_paths.append(f)
                    else:
                        full_p = os.path.join(dest_dir, f)
                        if os.path.exists(full_p):
                            downloaded_paths.append(full_p)
        else:
            dest_path = os.path.join(dest_dir, "resume.pdf")
            success = download_gdrive_file_safe(url, dest_path)
            if success:
                downloaded_paths.append(dest_path)
    except Exception as e:
        # Fallback regex parsing
        folder_match = re.search(r'/folders/([a-zA-Z0-9-_]+)', url)
        file_match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        try:
            if folder_match:
                folder_id = folder_match.group(1)
                folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
                files = gdown.download_folder(folder_url, output=dest_dir, quiet=True, remaining_ok=True, use_cookies=False)
                if files:
                    for f in files:
                        if os.path.exists(f):
                            downloaded_paths.append(f)
                        else:
                            full_p = os.path.join(dest_dir, f)
                            if os.path.exists(full_p):
                                downloaded_paths.append(full_p)
            elif file_match:
                file_id = file_match.group(1)
                dest_path = os.path.join(dest_dir, "resume.pdf")
                file_url = f"https://drive.google.com/uc?id={file_id}"
                success = download_gdrive_file_safe(file_url, dest_path)
                if success:
                    downloaded_paths.append(dest_path)
        except Exception as inner_e:
            raise RuntimeError(f"Failed to download from GDrive link: {str(inner_e)}") from e
            
    # Find all downloaded PDF files recursively
    pdf_files = []
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
                
    return pdf_files

