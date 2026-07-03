import json
import math
import os
import streamlit as st
import pandas as pd
import gdown
from pypdf import PdfReader
import io
import shutil
import tempfile
import threading
import time

from utils import extract_resume_data, compute_score, download_from_gdrive, generate_score_explanation, download_gdrive_file_safe

st.set_page_config(
    page_title="CS Resume Scorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #16a34a, #4ade80);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 10px; padding: 0.5rem 1rem;
    }

    /* ── Flag boxes ── */
    .flag-box {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 10px; padding: 0.55rem 1rem;
        margin: 0.25rem 0; font-size: 0.9rem; color: #14532d;
    }

    /* ── Formula box ── */
    .formula-box {
        background: #0f172a; color: #a3e635;
        border-left: 4px solid #16a34a; border-radius: 8px;
        padding: 0.75rem 1.1rem; font-family: 'Courier New', monospace;
        font-size: 0.85rem; margin: 0.5rem 0 0.8rem 0;
        white-space: pre-wrap; line-height: 1.6;
    }

    /* ── Step row ── */
    .step-row {
        display: flex; align-items: flex-start; gap: 12px;
        background: #f8fffe; border: 1px solid #d1fae5;
        border-radius: 8px; padding: 0.6rem 0.9rem; margin: 0.3rem 0;
    }
    .step-num {
        background: #16a34a; color: white; border-radius: 50%;
        width: 22px; height: 22px; display: flex; align-items: center;
        justify-content: center; font-size: 0.72rem; font-weight: 700;
        flex-shrink: 0; margin-top: 2px;
    }
    .step-text { font-size: 0.88rem; color: #1e3a2f; line-height: 1.5; }
    .step-val  { font-family: monospace; font-weight: 700; color: #16a34a; }

    /* ── Weight badge ── */
    .w-badge {
        display: inline-block; background: #dcfce7; color: #14532d;
        border: 1px solid #86efac; border-radius: 20px;
        padding: 0.1rem 0.55rem; font-size: 0.78rem; font-weight: 700;
        margin-left: 6px; vertical-align: middle;
    }

    /* ── Control panel ── */
    .ctrl-header {
        font-size: 0.78rem; font-weight: 700; color: #6b7280;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
    }

    /* ── Score card ── */
    .score-ring {
        text-align: center; border-radius: 16px; padding: 2rem 1rem;
        border-width: 2px; border-style: solid;
    }

    /* ── Tier chip ── */
    .tier1 { background:#fef9c3; color:#713f12; border:1px solid #fde047;
              border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
    .tier2 { background:#dbeafe; color:#1e3a8a; border:1px solid #93c5fd;
              border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
    .tier3 { background:#dcfce7; color:#14532d; border:1px solid #86efac;
              border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

#  SIDEBAR

env_openai_key = os.environ.get("OPENAI_API_KEY", "")
env_groq_key = os.environ.get("GROQ_API_KEY", "")
env_azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
env_azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
env_azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
env_claude_key = os.environ.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.markdown("## 🎓 CS Resume Scorer")
    st.divider()
    
    st.markdown("### ⚙️ LLM Settings")
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic Claude", "Azure OpenAI", "Custom OpenAI Proxy", "Groq"], index=0)
    
    azure_endpoint = None
    azure_deployment = None
    azure_api_version = "2024-02-15-preview"
    custom_base_url = None
    llm_model = None

    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", value=env_openai_key, placeholder="sk-...")
        llm_model_select = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-4-turbo", "gpt-4", "Custom Model"], index=0)
        if llm_model_select == "Custom Model":
            llm_model = st.text_input("Enter Model Name", placeholder="e.g. gpt-5")
        else:
            llm_model = llm_model_select
    elif llm_provider == "Anthropic Claude":
        api_key = st.text_input("Anthropic API Key", type="password", value=env_claude_key, placeholder="sk-ant-...")
        llm_model_select = st.selectbox("LLM Model", ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "Custom Model"], index=0)
        if llm_model_select == "Custom Model":
            llm_model = st.text_input("Enter Model Name", placeholder="e.g. claude-3-5-sonnet-20241022")
        else:
            llm_model = llm_model_select
    elif llm_provider == "Azure OpenAI":
        api_key = st.text_input("Azure API Key", type="password", value=env_azure_key, placeholder="Azure API Key")
        azure_endpoint = st.text_input("Azure Endpoint URL", value=env_azure_endpoint, placeholder="https://your-resource.openai.azure.com/")
        azure_deployment = st.text_input("Deployment Name", value=env_azure_deployment, placeholder="e.g. gpt-5-deployment")
        azure_api_version = st.text_input("API Version", value="2024-02-15-preview", placeholder="e.g. 2024-02-15-preview")
    elif llm_provider == "Custom OpenAI Proxy":
        api_key = st.text_input("Proxy API Key", type="password", placeholder="Paste API Key here...")
        custom_base_url = st.text_input("Proxy Base URL", placeholder="https://endpoint-domain/v1")
        llm_model_select = st.selectbox("LLM Model", ["gpt-5", "gpt-4o", "Custom Model"], index=0)
        if llm_model_select == "Custom Model":
            llm_model = st.text_input("Enter Model Name", placeholder="e.g. gpt-5")
        else:
            llm_model = llm_model_select
    else:
        api_key = st.text_input("Groq API Key", type="password", value=env_groq_key, placeholder="gsk_...")
        llm_model_select = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it", "Custom Model"], index=0)
        if llm_model_select == "Custom Model":
            llm_model = st.text_input("Enter Model Name", placeholder="e.g. llama-3.1-405b-instruct")
        else:
            llm_model = llm_model_select
        
    st.divider()
    
    input_source = st.radio("Input Source", ["Upload Single PDF", "Google Drive Link", "Upload Student Excel/CSV"], index=0)
    
    uploaded_file = None
    drive_link = ""
    uploaded_sheet = None
    
    if input_source == "Upload Single PDF":
        uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    elif input_source == "Google Drive Link":
        drive_link = st.text_input("Paste Google Drive Link", placeholder="https://drive.google.com/...")
    else:
        uploaded_sheet = st.file_uploader("Upload Student Sheet (Excel/CSV)", type=["xlsx", "xls", "csv"])
        
    st.divider()
    
    dashboard = st.radio(
        "View",
        ["📊 Score", "🔬 Formula Steps", "🎛️ Controls", "🗂️ Raw Data"],
        index=0,
    )
    st.divider()

#  CONSTANTS (overrideable from Controls tab)

DEFAULT_WEIGHTS = {
    2: {"hyg":0.25,"real":0.25,"comp":0.20,"imp":0.05,"prod":0.10,"clar":0.05,"dom":0.05,"vel":0.05},
    3: {"hyg":0.15,"real":0.20,"comp":0.25,"imp":0.10,"prod":0.15,"clar":0.05,"dom":0.05,"vel":0.05},
    4: {"hyg":0.05,"real":0.10,"comp":0.30,"imp":0.20,"prod":0.15,"clar":0.05,"dom":0.05,"vel":0.10},
}
DEFAULT_CONSTANTS = {
    "alpha": 5.0,    # S_complexity volume bonus
    "beta": 12.0,    # S_impact saturation multiplier
    "omega": 15.0,   # S_clarity buzzword penalty
    "eps": 1.0,      # division-by-zero guard
    "tier1_c": 25,   # project tier scores
    "tier2_c": 65,
    "tier3_c": 100,
    "hygiene_page_pen": 50,
    "hygiene_link_pen": 15,
    "hygiene_email_pen": 25,
    "hygiene_sec_pen": 20,
}

def get_overrides():
    return st.session_state.get("overrides", {
        "weights": {yr: dict(w) for yr, w in DEFAULT_WEIGHTS.items()},
        "constants": dict(DEFAULT_CONSTANTS),
    })

# ── Background Batch Processing System ──
BATCH_DIR = os.path.join(os.getcwd(), "data_batches")
os.makedirs(BATCH_DIR, exist_ok=True)
STATUS_FILE = os.path.join(BATCH_DIR, "status.json")
RESULTS_FILE = os.path.join(BATCH_DIR, "results.csv")
DETAILS_FILE = os.path.join(BATCH_DIR, "details.json")
INPUT_FILE = os.path.join(BATCH_DIR, "input_sheet.xlsx")

def save_batch_status(status_dict):
    with open(STATUS_FILE, "w") as f:
        json.dump(status_dict, f, indent=4)

def load_batch_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def is_worker_active() -> bool:
    for t in threading.enumerate():
        if t.name == "resugent_batch_worker":
            return True
    return False

def run_background_batch(name_col, link_col, api_key, llm_provider, llm_model, azure_endpoint, azure_deployment, azure_api_version, custom_base_url, overrides):
    try:
        # Load input sheet
        if INPUT_FILE.endswith(".csv") or not os.path.exists(INPUT_FILE):
            csv_path = INPUT_FILE.replace(".xlsx", ".csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            elif os.path.exists(INPUT_FILE):
                df = pd.read_excel(INPUT_FILE)
            else:
                raise FileNotFoundError("Input sheet file not found on disk.")
        else:
            df = pd.read_excel(INPUT_FILE)
            
        total_rows = len(df)
        
        # Load existing results if any (for resumption)
        processed_links = set()
        results = []
        
        if os.path.exists(RESULTS_FILE):
            try:
                df_existing = pd.read_csv(RESULTS_FILE)
                results = df_existing.to_dict(orient="records")
                processed_links = set(str(row["Resume Link"]).strip() for row in results)
            except Exception:
                pass

        # Temp download dir
        TEMP_SHEET_DIR = os.path.join(tempfile.gettempdir(), "resugent_sheet_downloads")
        os.makedirs(TEMP_SHEET_DIR, exist_ok=True)
        
        for idx, row in df.iterrows():
            # Check cancel flag
            current_status = load_batch_status()
            if current_status and current_status.get("status") in ["cancelling", "cancelled"]:
                current_status["status"] = "cancelled"
                save_batch_status(current_status)
                return
                
            student_name = str(row[name_col]).strip()
            resume_url = str(row[link_col]).strip()
            
            # Skip if already processed in previous run
            if resume_url in processed_links:
                continue
                
            # Update status
            status_data = load_batch_status()
            if status_data:
                status_data["processed"] = idx
                status_data["current_name"] = student_name
                save_batch_status(status_data)
                
            score_data = {"final_score": 0.0, "scores": {}}
            status = "failed"
            error_msg = ""
            explanation = ""
            resume_data_dict = None
            raw_text = ""
            
            if not resume_url or resume_url.lower() in ["nan", "none", "null", ""]:
                error_msg = "Empty resume link"
                explanation = "No resume link was provided in the Excel sheet."
            else:
                try:
                    dest_path = os.path.join(TEMP_SHEET_DIR, f"resume_{idx}.pdf")
                    success = download_gdrive_file_safe(resume_url, dest_path)
                    
                    if success and os.path.exists(dest_path):
                        with open(dest_path, "rb") as f:
                            file_bytes = f.read()
                        raw_text = extract_pdf_text(file_bytes)
                        if not raw_text:
                            error_msg = "Could not read PDF text"
                            explanation = "The PDF downloaded successfully but contains no readable text."
                        else:
                            resume_data = extract_resume_data(
                                raw_text, 
                                api_key, 
                                provider=llm_provider, 
                                model=llm_model,
                                azure_endpoint=azure_endpoint,
                                azure_deployment=azure_deployment,
                                azure_api_version=azure_api_version,
                                base_url=custom_base_url
                            )
                            # Convert to dict for serialization
                            resume_data_dict = {
                                "candidate_name": resume_data.candidate_name,
                                "total_page_count": resume_data.total_page_count,
                                "extracted_links_array": resume_data.extracted_links_array,
                                "raw_email_string": resume_data.raw_email_string,
                                "detected_section_headers": resume_data.detected_section_headers,
                                "project_count": resume_data.project_count,
                                "project_titles": resume_data.project_titles,
                                "project_tech_keywords": resume_data.project_tech_keywords,
                                "architectural_regex_flags": resume_data.architectural_regex_flags,
                                "deployment_live_urls": resume_data.deployment_live_urls,
                                "code_repository_urls": resume_data.code_repository_urls,
                                "regex_extracted_numeric_values": resume_data.regex_extracted_numeric_values,
                                "metric_regex_match_count": resume_data.metric_regex_match_count,
                                "buzzword_frequency_map": resume_data.buzzword_frequency_map,
                                "skills_section_keywords": resume_data.skills_section_keywords,
                                "domain_classification_vector": resume_data.domain_classification_vector,
                                "experience_timeline_intervals": resume_data.experience_timeline_intervals,
                                "btech_year": resume_data.btech_year
                            }
                            score_data = compute_score_custom(resume_data, overrides)
                            try:
                                explanation = generate_score_explanation(
                                    raw_text, 
                                    score_data, 
                                    api_key, 
                                    provider=llm_provider, 
                                    model=llm_model,
                                    azure_endpoint=azure_endpoint,
                                    azure_deployment=azure_deployment,
                                    azure_api_version=azure_api_version,
                                    base_url=custom_base_url
                                )
                            except Exception as e_exp:
                                explanation = f"Could not generate feedback: {str(e_exp)}"
                            status = "success"
                    else:
                        error_msg = "Failed to download / Private link"
                        explanation = "Resume PDF could not be downloaded (Check link sharing settings)."
                except Exception as ex:
                    error_msg = str(ex)
                    explanation = f"Failed to download/process resume: {str(ex)} (Check link sharing settings)."
            
            # Map columns
            row_res = {
                "Student Name": student_name,
                "Resume Link": resume_url,
                "Status": "✅ Success" if status == "success" else "❌ Failed",
                "Error Details": error_msg,
                "Score": score_data.get("final_score", 0.0),
                "S_hygiene": score_data.get("scores", {}).get("S_hygiene", 0.0) if status == "success" else 0.0,
                "S_realization": score_data.get("scores", {}).get("S_realization", 0.0) if status == "success" else 0.0,
                "S_complexity": score_data.get("scores", {}).get("S_complexity", 0.0) if status == "success" else 0.0,
                "S_impact": score_data.get("scores", {}).get("S_impact", 0.0) if status == "success" else 0.0,
                "S_production": score_data.get("scores", {}).get("S_production", 0.0) if status == "success" else 0.0,
                "S_clarity": score_data.get("scores", {}).get("S_clarity", 0.0) if status == "success" else 0.0,
                "S_domain": score_data.get("scores", {}).get("S_domain", 0.0) if status == "success" else 0.0,
                "S_velocity": score_data.get("scores", {}).get("S_velocity", 0.0) if status == "success" else 0.0,
                "Explanation": explanation
            }
            results.append(row_res)
            
            df_results = pd.DataFrame(results)
            df_results.to_csv(RESULTS_FILE, index=False)
            
            if status == "success" and resume_data_dict:
                current_details = {}
                if os.path.exists(DETAILS_FILE):
                    try:
                        with open(DETAILS_FILE, "r") as f:
                            current_details = json.load(f)
                    except Exception:
                        pass
                current_details[student_name] = {
                    "resume_data": resume_data_dict,
                    "score_data": score_data,
                    "raw_text": raw_text,
                    "explanation": explanation
                }
                with open(DETAILS_FILE, "w") as f:
                    json.dump(current_details, f, indent=4)
                    
            status_data = load_batch_status()
            if status_data:
                status_data["processed"] = idx + 1
                if status == "success":
                    status_data["success_count"] += 1
                else:
                    status_data["failed_count"] += 1
                save_batch_status(status_data)
                
        status_data = load_batch_status()
        if status_data:
            status_data["status"] = "completed"
            status_data["current_name"] = ""
            save_batch_status(status_data)
            
    except Exception as e_run:
        status_data = load_batch_status()
        if status_data:
            status_data["status"] = "error"
            status_data["error_msg"] = str(e_run)
            save_batch_status(status_data)

def check_and_resume_batch():
    status_data = load_batch_status()
    if status_data and status_data.get("status") == "running" and not is_worker_active():
        cfg = status_data.get("config", {})
        if cfg:
            t = threading.Thread(
                target=run_background_batch,
                args=(
                    cfg.get("name_col"),
                    cfg.get("link_col"),
                    cfg.get("api_key"),
                    cfg.get("llm_provider"),
                    cfg.get("llm_model"),
                    cfg.get("azure_endpoint"),
                    cfg.get("azure_deployment"),
                    cfg.get("azure_api_version"),
                    cfg.get("custom_base_url"),
                    cfg.get("overrides")
                ),
                name="resugent_batch_worker",
                daemon=True
            )
            t.start()

check_and_resume_batch()

#  SCORING ENGINE (uses overrides)
# ─────────────────────────────────────────────
TIER3_SKILLS = {"golang","go","docker","kubernetes","redis","kafka","grpc","aws","gcp","azure",
                "tensorflow","pytorch","spark","hadoop","elasticsearch","rabbitmq","celery",
                "websockets","microservices","ci/cd","jenkins","terraform"}
TIER2_SKILLS = {"python","java","javascript","typescript","react","nodejs","node.js","sql",
                "mongodb","postgresql","mysql","git","spring","fastapi","flask","django",
                "express","graphql","rest","linux","bash","c#","kotlin","swift"}

def skill_difficulty(skill):
    s = skill.lower().strip()
    if s in TIER3_SKILLS: return 10
    if s in TIER2_SKILLS: return 5
    return 2

def project_tier(tech_keywords, arch_flag, C):
    if arch_flag: return C["tier3_c"]
    kw = {k.lower() for k in tech_keywords}
    if kw & TIER3_SKILLS: return C["tier3_c"]
    has_backend = bool(kw & {"nodejs","node.js","express","django","flask","fastapi","spring","java","python","golang"})
    has_db = bool(kw & {"mongodb","postgresql","mysql","sql","redis","firebase","supabase"})
    if has_backend and has_db: return C["tier2_c"]
    return C["tier1_c"]

def compute_score_custom(data, overrides):
    C = overrides["constants"]
    year = data.btech_year if data.btech_year in overrides["weights"] else 3
    W = overrides["weights"][year]
    eps = C["eps"]
    steps = {}   

    # ── 1. S_hygiene ──
    P = max(data.total_page_count, 1)
    links_lower = [l.lower() for l in data.extracted_links_array]
    has_github = any("github" in l for l in links_lower)
    has_linkedin = any("linkedin" in l for l in links_lower)
    L_missing = (0 if has_github else 1) + (0 if has_linkedin else 1)
    email = data.raw_email_string.lower()
    E_generic = 1 if (any(c.isdigit() for c in email.split("@")[0]) or
                      any(w in email for w in ["cool","coder","gamer","noob","pro","god","king","boss"])) else 0
    mandatory = {"education", "projects", "skills"}
    found = {h.lower() for h in data.detected_section_headers}
    X_missing = len(mandatory - found)
    S_hygiene = max(0, 100
        - C["hygiene_page_pen"] * max(0, P - 1)
        - C["hygiene_link_pen"] * L_missing
        - C["hygiene_email_pen"] * E_generic
        - C["hygiene_sec_pen"] * X_missing)
    steps["hygiene"] = {
        "P": P, "L_missing": L_missing, "E_generic": E_generic,
        "X_missing": X_missing, "score": round(S_hygiene, 2),
        "has_github": has_github, "has_linkedin": has_linkedin,
    }

    # ── 2. S_realization ──
    declared = set(k.lower().strip() for k in data.skills_section_keywords)
    corpus = (data.project_descriptions_text_corpus + " " + data.experience_descriptions_text_corpus).lower()
    applied = {k for k in declared if k in corpus}
    intersect = declared & applied
    sum_intersect = sum(math.log(skill_difficulty(k) + 1) for k in intersect)
    sum_declared  = sum(math.log(skill_difficulty(k) + 1) for k in declared) + eps
    S_realization = (sum_intersect / sum_declared) * 100
    steps["realization"] = {
        "declared_count": len(declared),
        "applied_count": len(applied),
        "intersect_count": len(intersect),
        "sum_intersect": round(sum_intersect, 3),
        "sum_declared": round(sum_declared, 3),
        "score": round(S_realization, 2),
        "unverified": list(declared - applied)[:8],
        "verified": list(intersect)[:8],
    }

    # ── 3. S_complexity ──
    alpha = C["alpha"]
    tiers = []
    if data.project_titles:
        for i, _ in enumerate(data.project_titles):
            tech = data.project_tech_keywords[i] if i < len(data.project_tech_keywords) else []
            arch = data.architectural_regex_flags[i] if i < len(data.architectural_regex_flags) else False
            tiers.append(project_tier(tech, arch, C))
        max_cj = max(tiers)
        J = len(data.project_titles)
        S_complexity = min(100, max_cj + alpha * math.log(J + 1))
    else:
        max_cj, J, S_complexity = 0, 0, 0.0
    steps["complexity"] = {
        "project_tiers": list(zip(data.project_titles, tiers)),
        "max_cj": max_cj, "J": J, "alpha": alpha,
        "log_bonus": round(alpha * math.log(J + 1) if J else 0, 3),
        "score": round(S_complexity, 2),
    }

    # ── 4. S_impact ──
    beta = C["beta"]
    values = data.regex_extracted_numeric_values or []
    log_sum = sum(math.log10(v + 1) for v in values if v > 0)
    S_impact = min(100, beta * log_sum)
    steps["impact"] = {
        "values": values[:10], "beta": beta,
        "log_sum": round(log_sum, 3), "score": round(S_impact, 2),
    }

    # ── 5. S_production ──
    J_total = max(data.project_count, 1)
    J_code = len(data.code_repository_urls)
    J_deploy = len(data.deployment_live_urls)
    S_production = ((J_code + J_deploy) / (2 * J_total)) * 100
    steps["production"] = {
        "J": J_total, "J_code": J_code, "J_deploy": J_deploy,
        "score": round(S_production, 2),
    }

    # ── 6. S_clarity ──
    omega = C["omega"]
    bmap = data.buzzword_frequency_map or {}
    deduction = omega * sum(math.log(cnt + 1) for cnt in bmap.values() if cnt > 0)
    S_clarity = max(0, 100 - deduction)
    steps["clarity"] = {
        "buzzwords": bmap, "omega": omega,
        "deduction": round(deduction, 3), "score": round(S_clarity, 2),
    }

    # ── 7. S_domain ──
    unique_domains = len(set(data.domain_classification_vector))
    total_skills = len(data.skills_section_keywords) + eps
    S_domain = max(0, min(100, 100 * (1 - unique_domains / total_skills)))
    steps["domain"] = {
        "unique_domains": unique_domains, "total_skills": len(data.skills_section_keywords),
        "domains": list(set(data.domain_classification_vector)),
        "score": round(S_domain, 2),
    }

    # ── 8. S_velocity ──
    role_weights = {"internship": 15, "freelance": 10, "tech_lead": 10, "member": 3}
    velocity_sum = sum(
        e.get("months", 0) * role_weights.get(e.get("type", "member"), 3)
        for e in data.experience_timeline_intervals
    )
    S_velocity = min(100, velocity_sum)
    steps["velocity"] = {
        "entries": data.experience_timeline_intervals[:6],
        "role_weights": role_weights,
        "raw_sum": round(velocity_sum, 2),
        "score": round(S_velocity, 2),
    }

    # ── Final ──
    scores = {
        "S_hygiene": S_hygiene, "S_realization": S_realization,
        "S_complexity": S_complexity, "S_impact": S_impact,
        "S_production": S_production, "S_clarity": S_clarity,
        "S_domain": S_domain, "S_velocity": S_velocity,
    }
    keys = ["hyg","real","comp","imp","prod","clar","dom","vel"]
    skeys = list(scores.keys())
    final = sum(W[keys[i]] * list(scores.values())[i] for i in range(8))
    final = round(min(100, max(0, final)), 2)

    return {
        "final_score": final, "btech_year": year, "weights": W,
        "scores": scores, "steps": steps,
        "L_missing": L_missing, "E_generic": E_generic, "X_missing": X_missing,
        "buzzwords_found": bmap,
        **{k: round(v, 2) for k, v in scores.items()},
    }

#  PDF + PIPELINE
# ─────────────────────────────────────────────
def extract_pdf_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()

def run_pipeline(file_bytes, api_key, provider="openai", model=None, azure_endpoint=None, azure_deployment=None, azure_api_version="2024-02-15-preview", base_url=None):
    with st.spinner("Reading PDF..."):
        try:
            raw_text = extract_pdf_text(file_bytes)
        except Exception as e:
            st.error(f"Could not read PDF: {str(e)}")
            return False
    if not raw_text:
        st.error("Could not read PDF — make sure it's not a scanned image.")
        return False
    with st.spinner("LLM extracting resume fields..."):
        try:
            resume_data = extract_resume_data(
                raw_text, 
                api_key, 
                provider=provider, 
                model=model, 
                azure_endpoint=azure_endpoint, 
                azure_deployment=azure_deployment, 
                azure_api_version=azure_api_version,
                base_url=base_url
            )
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e) or "invalid_api_key" in str(e):
                st.error("⚠️ Authentication Error: The provided LLM API Key is incorrect or invalid. Please check your key in the sidebar.")
            elif "RateLimitError" in type(e).__name__ or "429" in str(e):
                st.error("⚠️ Rate Limit Error: You have exceeded the LLM provider rate limit. Please try again in a moment.")
            else:
                st.error(f"⚠️ LLM Extraction failed: {str(e)}")
            return False
    with st.spinner("Computing scores..."):
        try:
            score_data = compute_score_custom(resume_data, get_overrides())
        except Exception as e:
            st.error(f"⚠️ Score computation failed: {str(e)}")
            return False
    with st.spinner("Generating explainable review..."):
        try:
            explanation = generate_score_explanation(
                raw_text, 
                score_data, 
                api_key, 
                provider=provider, 
                model=model,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                azure_api_version=azure_api_version,
                base_url=base_url
            )
        except Exception as e:
            explanation = f"Could not generate explainable feedback: {str(e)}"
    st.session_state.update({
        "raw_text": raw_text, 
        "resume_data": resume_data, 
        "score_data": score_data,
        "score_explanation": explanation
    })
    return True

#  HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">🎓 CS Resume Scorer</p>', unsafe_allow_html=True)
st.caption("8-dimension scoring engine · LLM extraction · adjustable weights & constants")

TEMP_DIR = os.path.join(tempfile.gettempdir(), "resugent_gdrive_downloads")

if input_source == "Upload Single PDF":
    if "bulk_results" in st.session_state:
        del st.session_state["bulk_results"]
    if "sheet_results" in st.session_state:
        del st.session_state["sheet_results"]
        
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_id = hash(file_bytes)
        if st.session_state.get("processed_file") != file_id or "resume_data" not in st.session_state:
            if not api_key:
                st.error("⚠️ Please provide the LLM API Key in the sidebar.")
                st.stop()
            if run_pipeline(
                file_bytes, 
                api_key, 
                provider=llm_provider, 
                model=llm_model,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                azure_api_version=azure_api_version,
                base_url=custom_base_url
            ):
                st.session_state["processed_file"] = file_id
            else:
                if "processed_file" in st.session_state:
                    del st.session_state["processed_file"]
                st.stop()
    else:
        if "resume_data" not in st.session_state:
            st.info("👈 Upload a resume PDF to get started.")
            st.stop()

elif input_source == "Google Drive Link":
    if "sheet_results" in st.session_state:
        del st.session_state["sheet_results"]
        
    if st.sidebar.button("🚀 Fetch & Process Resumes", key="process_gdrive_btn"):
        if not api_key:
            st.sidebar.error("⚠️ Please provide the LLM API Key in the sidebar.")
            st.stop()
        if not drive_link:
            st.sidebar.error("⚠️ Please paste a Google Drive folder or file link.")
            st.stop()
            
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        with st.spinner("Downloading resumes from Google Drive..."):
            try:
                pdf_files = download_from_gdrive(drive_link, TEMP_DIR)
            except Exception as e:
                st.sidebar.error(f"Failed to download files: {str(e)}")
                st.stop()
                
        if not pdf_files:
            st.sidebar.error("No PDF files found in the Google Drive link. Please make sure the folder is shared publicly.")
            st.stop()
            
        bulk_results = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        for idx, filepath in enumerate(pdf_files):
            filename = os.path.basename(filepath)
            status_text.text(f"Processing ({idx+1}/{len(pdf_files)}): {filename}...")
            progress_bar.progress(idx / len(pdf_files))
            
            try:
                with open(filepath, "rb") as f:
                    file_bytes = f.read()
                raw_text = extract_pdf_text(file_bytes)
                if not raw_text:
                    bulk_results.append({
                        "filename": filename,
                        "candidate_name": "Unknown (Empty text)",
                        "final_score": 0.0,
                        "status": "failed",
                        "error": "Could not read PDF text"
                    })
                    continue
                
                resume_data = extract_resume_data(
                    raw_text, 
                    api_key, 
                    provider=llm_provider, 
                    model=llm_model,
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                    azure_api_version=azure_api_version,
                    base_url=custom_base_url
                )
                score_data = compute_score_custom(resume_data, get_overrides())
                try:
                    explanation = generate_score_explanation(
                        raw_text, 
                        score_data, 
                        api_key, 
                        provider=llm_provider, 
                        model=llm_model,
                        azure_endpoint=azure_endpoint,
                        azure_deployment=azure_deployment,
                        azure_api_version=azure_api_version,
                        base_url=custom_base_url
                    )
                except Exception as e_exp:
                    explanation = f"Could not generate feedback: {str(e_exp)}"
                
                bulk_results.append({
                    "filename": filename,
                    "candidate_name": resume_data.candidate_name,
                    "final_score": score_data["final_score"],
                    "status": "success",
                    "resume_data": resume_data,
                    "score_data": score_data,
                    "raw_text": raw_text,
                    "explanation": explanation
                })
            except Exception as ex:
                bulk_results.append({
                    "filename": filename,
                    "candidate_name": "Unknown (Processing error)",
                    "final_score": 0.0,
                    "status": "failed",
                    "error": str(ex)
                })
                
        progress_bar.progress(1.0)
        status_text.text(f"Processed all {len(pdf_files)} resumes successfully!")
        
        st.session_state["bulk_results"] = bulk_results
        
        success_candidates = [c for c in bulk_results if c["status"] == "success"]
        if success_candidates:
            st.session_state["resume_data"] = success_candidates[0]["resume_data"]
            st.session_state["score_data"] = success_candidates[0]["score_data"]
            st.session_state["raw_text"] = success_candidates[0]["raw_text"]
            st.session_state["score_explanation"] = success_candidates[0]["explanation"]
            st.session_state["selected_bulk_candidate"] = success_candidates[0]["candidate_name"]
            
        st.rerun()

    if "bulk_results" in st.session_state:
        bulk_results = st.session_state["bulk_results"]
        
        st.markdown("## 📊 Bulk Processing Dashboard")
        
        total_resumes = len(bulk_results)
        success_resumes = sum(1 for c in bulk_results if c["status"] == "success")
        failed_resumes = total_resumes - success_resumes
        
        success_scores = [c["final_score"] for c in bulk_results if c["status"] == "success"]
        avg_score = round(sum(success_scores) / len(success_scores), 2) if success_scores else 0.0
        max_score = max(success_scores) if success_scores else 0.0
        
        top_cand = "-"
        if success_scores:
            top_cand_idx = success_scores.index(max_score)
            top_cand = [c for c in bulk_results if c["status"] == "success"][top_cand_idx]["candidate_name"]
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Resumes", f"{total_resumes} ({success_resumes} OK / {failed_resumes} Fail)")
        m2.metric("Average Score", f"{avg_score} / 100")
        m3.metric("Highest Score", f"{max_score} / 100")
        m4.metric("Top Candidate", f"{top_cand}")
        
        table_data = []
        for c in bulk_results:
            row = {
                "Filename": c["filename"],
                "Candidate Name": c["candidate_name"],
                "Status": "✅ Success" if c["status"] == "success" else f"❌ Failed: {c.get('error', 'Unknown Error')}",
                "B.Tech Year": c["resume_data"].btech_year if c["status"] == "success" else "-",
                "Score": c["final_score"] if c["status"] == "success" else 0.0,
            }
            if c["status"] == "success":
                sd = c["score_data"]
                row.update({
                    "S_hygiene": sd.get("S_hygiene", 0),
                    "S_realization": sd.get("S_realization", 0),
                    "S_complexity": sd.get("S_complexity", 0),
                    "S_impact": sd.get("S_impact", 0),
                    "S_production": sd.get("S_production", 0),
                    "S_clarity": sd.get("S_clarity", 0),
                    "S_domain": sd.get("S_domain", 0),
                    "S_velocity": sd.get("S_velocity", 0),
                })
            else:
                row.update({k: 0.0 for k in ["S_hygiene", "S_realization", "S_complexity", "S_impact", "S_production", "S_clarity", "S_domain", "S_velocity"]})
            table_data.append(row)
            
        st.markdown("### 📋 Resumes Summary Table")
        st.dataframe(table_data, use_container_width=True)
        
        st.markdown("---")
        
        success_names = [c["candidate_name"] for c in bulk_results if c["status"] == "success"]
        if success_names:
            st.markdown("### 🔍 Detailed Candidate Inspector")
            selected_name = st.selectbox(
                "Select a candidate to load their detailed 8-dimension report below:",
                options=success_names,
                index=success_names.index(st.session_state.get("selected_bulk_candidate", success_names[0])) if st.session_state.get("selected_bulk_candidate") in success_names else 0
            )
            if selected_name != st.session_state.get("selected_bulk_candidate"):
                cand_data = next(c for c in bulk_results if c["candidate_name"] == selected_name)
                st.session_state["resume_data"] = cand_data["resume_data"]
                st.session_state["score_data"] = cand_data["score_data"]
                st.session_state["raw_text"] = cand_data["raw_text"]
                st.session_state["score_explanation"] = cand_data["explanation"]
                st.session_state["selected_bulk_candidate"] = selected_name
                st.rerun()
        else:
            st.warning("No successfully processed resumes to inspect.")
            st.stop()
    else:
        st.info("👈 Paste a Google Drive link and click **Fetch & Process Resumes** in the sidebar to download and process resumes in bulk.")
        st.stop()

else:  # Upload Student Excel/CSV
    if "bulk_results" in st.session_state:
        del st.session_state["bulk_results"]
        
    status_data = load_batch_status()
    
    if status_data:
        batch_status = status_data.get("status")
        total = status_data.get("total", 0)
        processed = status_data.get("processed", 0)
        success_count = status_data.get("success_count", 0)
        failed_count = status_data.get("failed_count", 0)
        current_name = status_data.get("current_name", "")
        
        st.markdown("## 📊 Background Batch Processing Dashboard")
        
        df_results = None
        if os.path.exists(RESULTS_FILE):
            try:
                df_results = pd.read_csv(RESULTS_FILE)
            except Exception:
                pass
                
        # Status card UI
        if batch_status == "running":
            progress_pct = min(1.0, processed / max(1, total))
            st.info(f"⏳ **Active Batch Running in Background:** Processing candidate {processed}/{total} (Currently: `{current_name}`)")
            st.progress(progress_pct)
            
            c1, c2 = st.columns([1, 4])
            if c1.button("🛑 Cancel Batch"):
                status_data["status"] = "cancelling"
                save_batch_status(status_data)
                st.rerun()
            if c2.button("🔄 Refresh Status"):
                st.rerun()
                
        elif batch_status == "cancelling":
            st.warning("🛑 **Cancelling batch...** Please wait for the current candidate to finish.")
            if st.button("🔄 Refresh Status"):
                st.rerun()
                
        elif batch_status == "cancelled":
            st.error("🛑 **Batch Cancelled by User.**")
            if st.button("🗑️ Reset and Start New Batch"):
                try:
                    shutil.rmtree(BATCH_DIR)
                except Exception:
                    pass
                os.makedirs(BATCH_DIR, exist_ok=True)
                st.rerun()
                
        elif batch_status == "error":
            st.error(f"❌ **Batch Failed with Error:** {status_data.get('error_msg')}")
            if st.button("🗑️ Reset and Start New Batch"):
                try:
                    shutil.rmtree(BATCH_DIR)
                except Exception:
                    pass
                os.makedirs(BATCH_DIR, exist_ok=True)
                st.rerun()
                
        elif batch_status == "completed":
            st.success("✅ **Batch Processing Completed Successfully!**")
            if st.button("🗑️ Reset and Start New Batch"):
                try:
                    shutil.rmtree(BATCH_DIR)
                except Exception:
                    pass
                os.makedirs(BATCH_DIR, exist_ok=True)
                st.rerun()
                
        # Metrics & Summary Table
        if df_results is not None and len(df_results) > 0:
            total_rows = len(df_results)
            success_rows = sum(df_results["Status"].str.contains("Success", na=False))
            failed_rows = total_rows - success_rows
            avg_score = round(df_results[df_results["Status"].str.contains("Success", na=False)]["Score"].mean(), 2) if success_rows else 0.0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Processed So Far", f"{total_rows} / {total}")
            m2.metric("Success Count", f"{success_rows} OK")
            m3.metric("Failed Count", f"{failed_rows} Fail")
            m4.metric("Average Score", f"{avg_score} / 100")
            
            # Excel export bytes
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Scores')
            excel_data = output.getvalue()
            
            st.download_button(
                label="⬇️ Download Processed Excel Sheet",
                data=excel_data,
                file_name="student_resume_scores.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_btn"
            )
            
            st.markdown("### 📋 Results Summary Table")
            st.dataframe(df_results, use_container_width=True)
            
            if os.path.exists(DETAILS_FILE):
                try:
                    with open(DETAILS_FILE, "r") as f:
                        details_data = json.load(f)
                except Exception:
                    details_data = {}
                    
                success_names = list(details_data.keys())
                if success_names:
                    st.markdown("### 🔍 Detailed Candidate Inspector")
                    selected_name = st.selectbox(
                        "Select a candidate to load their detailed 8-dimension report below:",
                        options=success_names,
                        index=success_names.index(st.session_state.get("selected_bulk_candidate", success_names[0])) if st.session_state.get("selected_bulk_candidate") in success_names else 0
                    )
                    
                    if selected_name != st.session_state.get("selected_bulk_candidate") or "resume_data" not in st.session_state:
                        details = details_data[selected_name]
                        st.session_state["resume_data"] = ResumeData(**details["resume_data"])
                        st.session_state["score_data"] = details["score_data"]
                        st.session_state["raw_text"] = details["raw_text"]
                        st.session_state["score_explanation"] = details["explanation"]
                        st.session_state["selected_bulk_candidate"] = selected_name
                        st.rerun()
        else:
            st.warning("No rows processed yet.")
            
    else:
        if uploaded_sheet is not None:
            try:
                if uploaded_sheet.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_sheet)
                else:
                    df = pd.read_excel(uploaded_sheet)
            except Exception as e:
                st.error(f"Failed to read sheet: {str(e)}")
                st.stop()
                
            st.markdown("### 📋 Uploaded Sheet Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Auto-detect columns
            detected_name_col = next((c for c in df.columns if any(x in c.lower() for x in ["name", "student", "candidate"])), df.columns[0])
            detected_link_col = next((c for c in df.columns if any(x in c.lower() for x in ["link", "resume", "url", "drive", "path"])), df.columns[-1])
            
            st.markdown("### ⚙️ Column Selection Mapping")
            c1, c2 = st.columns(2)
            name_col = c1.selectbox("Select Student Name Column", df.columns, index=list(df.columns).index(detected_name_col))
            link_col = c2.selectbox("Select Resume Drive Link Column", df.columns, index=list(df.columns).index(detected_link_col))
            
            if st.button("🚀 Start Background Batch Processing", key="start_sheet_btn"):
                if not api_key:
                    st.error("⚠️ Please provide the LLM API Key in the sidebar.")
                    st.stop()
                    
                # Save input sheet
                if uploaded_sheet.name.endswith(".csv"):
                    df.to_csv(INPUT_FILE.replace(".xlsx", ".csv"), index=False)
                else:
                    df.to_excel(INPUT_FILE, index=False)
                    
                # Save initial status config
                status_dict = {
                    "status": "running",
                    "total": len(df),
                    "processed": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "current_name": "",
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "config": {
                        "name_col": name_col,
                        "link_col": link_col,
                        "api_key": api_key,
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                        "azure_endpoint": azure_endpoint,
                        "azure_deployment": azure_deployment,
                        "azure_api_version": azure_api_version,
                        "custom_base_url": custom_base_url,
                        "overrides": get_overrides()
                    }
                }
                save_batch_status(status_dict)
                
                # Start worker thread
                t = threading.Thread(
                    target=run_background_batch,
                    args=(
                        name_col, link_col, api_key, llm_provider, llm_model,
                        azure_endpoint, azure_deployment, azure_api_version,
                        custom_base_url, get_overrides()
                    ),
                    name="resugent_batch_worker",
                    daemon=True
                )
                t.start()
                
                st.rerun()
        else:
            st.info("👈 Upload a Student Excel/CSV sheet in the sidebar to start.")
            st.stop()

if "resume_data" not in st.session_state or "score_data" not in st.session_state:
    st.stop()

resume_data = st.session_state["resume_data"]
score_data  = st.session_state["score_data"]
overrides   = get_overrides()

COMP_META = [
    ("S_hygiene",     "🏗️ Structural Hygiene",    "hyg",  "Page count, links, email, section headers"),
    ("S_realization", "🔗 Tech Realization",       "real", "Skills declared vs. actually used in projects"),
    ("S_complexity",  "⚙️ Project Complexity",     "comp", "Best project tier + log volume bonus"),
    ("S_impact",      "📈 Quantifiable Impact",    "imp",  "Log-dampened magnitude of metric bullets"),
    ("S_production",  "🚀 Production Readiness",   "prod", "GitHub repos + live deploy links per project"),
    ("S_clarity",     "🔇 Linguistic Clarity",     "clar", "Penalises buzzwords; entropy-based"),
    ("S_domain",      "🎯 Domain Specialisation",  "dom",  "Rewards focus — fewer domains across more skills"),
    ("⏱️ Velocity",   "⏱️ Chrono Velocity",        "vel",  "Internship/freelance/club duration × role weight"),
]
# fix key list
COMP_KEYS = [
    ("S_hygiene","🏗️ Structural Hygiene","hyg","Page count, links, email, section headers"),
    ("S_realization","🔗 Tech Realization","real","Skills declared vs. actually used in projects"),
    ("S_complexity","⚙️ Project Complexity","comp","Best project tier + log volume bonus"),
    ("S_impact","📈 Quantifiable Impact","imp","Log-dampened magnitude of metric bullets"),
    ("S_production","🚀 Production Readiness","prod","GitHub repos + live deploy links per project"),
    ("S_clarity","🔇 Linguistic Clarity","clar","Penalises buzzwords; entropy-based"),
    ("S_domain","🎯 Domain Specialisation","dom","Rewards focus — fewer domains across more skills"),
    ("S_velocity","⏱️ Chrono Velocity","vel","Internship/freelance/club duration × role weight"),
]

W = score_data.get("weights", overrides["weights"].get(score_data.get("btech_year", 3), {}))

#  TAB: 
# ═══════════════════════════════════════════════
if dashboard == "📊 Score":
    final_score = score_data.get("final_score", 0)
    btech_year  = score_data.get("btech_year", 3)
    sc = "#16a34a" if final_score>=75 else ("#4ade80" if final_score>=55 else ("#f59e0b" if final_score>=35 else "#ef4444"))
    grade = ("🏆 Strong Profile" if final_score>=75 else ("✅ Good Profile" if final_score>=55
             else ("⚠️ Average Profile" if final_score>=35 else "❌ Needs Work")))

    st.markdown("---")
    col_score, col_info, col_flags = st.columns([1.2, 1.5, 1.5])

    with col_score:
        st.markdown(f"""
        <div class="score-ring" style="background:#f0fdf4; border-color:{sc};">
            <div style="font-size:0.85rem;color:#4b5563;letter-spacing:.1em;text-transform:uppercase;">Final Score</div>
            <div style="font-size:4rem;font-weight:900;color:{sc};line-height:1.1;">{final_score}</div>
            <div style="font-size:0.8rem;color:#6b7280;">/ 100</div>
            <div style="margin-top:.5rem;font-size:1rem;font-weight:600;color:{sc};">{grade}</div>
            <div style="margin-top:.3rem;font-size:0.78rem;color:#6b7280;">Year {btech_year} weight matrix</div>
        </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown("#### 👤 Candidate")
        st.metric("Name", resume_data.candidate_name)
        st.metric("B.Tech Year", f"Year {btech_year}")
        st.metric("Projects", resume_data.project_count)
        st.metric("Skills Listed", len(resume_data.skills_section_keywords))

    with col_flags:
        st.markdown("#### 🔍 Quick Signals")
        flags = [
            ("GitHub link present",    any("github"   in l.lower() for l in resume_data.extracted_links_array)),
            ("LinkedIn link present",  any("linkedin" in l.lower() for l in resume_data.extracted_links_array)),
            ("Professional email",     score_data.get("E_generic", 0) == 0),
            ("Single-page resume",     resume_data.total_page_count == 1),
            ("All sections detected",  score_data.get("X_missing", 0) == 0),
            ("Live deployments found", len(resume_data.deployment_live_urls) > 0),
            ("Metric bullets present", resume_data.metric_regex_match_count > 0),
            ("Zero buzzwords",         len(resume_data.buzzword_frequency_map) == 0),
        ]
        for label, ok in flags:
            bg = "#dcfce7" if ok else "#fef2f2"
            st.markdown(f'<div class="flag-box" style="background:{bg};">{"✅" if ok else "❌"} {label}</div>', unsafe_allow_html=True)

    # ── Explainability Feedback ──
    if st.session_state.get("score_explanation"):
        st.markdown("---")
        st.markdown("### 💡 LLM Feedback & Suggestions")
        st.info(st.session_state["score_explanation"])

    # ── Score breakdown table ──
    st.markdown("---")
    st.markdown("### 📋 Score Breakdown")

    # weight sum sanity
    w_sum = sum(W.get(wk, 0) for _, _, wk, _ in COMP_KEYS)

    for skey, label, wkey, hint in COMP_KEYS:
        raw      = score_data.get(skey, 0)
        weight   = W.get(wkey, 0)
        weighted = round(raw * weight, 2)
        pct      = min(raw / 100, 1.0)

        c1, c2, c3, c4 = st.columns([3.2, 0.7, 0.7, 0.7])
        with c1:
            st.markdown(f"**{label}** <span class='w-badge'>w = {int(weight*100)}%</span>", unsafe_allow_html=True)
            st.caption(hint)
            color = "#16a34a" if raw >= 70 else ("#f59e0b" if raw >= 40 else "#ef4444")
            st.progress(pct)
        with c2:
            st.metric("Raw /100", f"{raw}")
        with c3:
            st.metric("Weight", f"{int(weight*100)}%")
        with c4:
            st.metric("Contribution", f"{weighted}")
        st.markdown("")

    # ── Final computation row ──
    st.markdown("---")
    st.markdown("### 🧮 Final = Σ (weight × score)")

    cols = st.columns(9)
    total_check = 0
    for i, (skey, label, wkey, _) in enumerate(COMP_KEYS):
        raw    = score_data.get(skey, 0)
        weight = W.get(wkey, 0)
        contrib = round(raw * weight, 2)
        total_check += contrib
        short = label.split()[1]
        cols[i].metric(short, f"{contrib}", delta=f"w={int(weight*100)}%", delta_color="off")
    cols[8].metric("🏁 Final", f"{final_score}")

    # ── Warnings ──
    st.markdown("---")
    if score_data.get("L_missing", 0) > 0:
        st.warning(f"⚠️ {score_data.get('L_missing')} link(s) missing — add GitHub and LinkedIn.")
    if score_data.get("E_generic", 0) == 1:
        st.warning("⚠️ Email appears unprofessional — use firstname.lastname@gmail.com.")
    if score_data.get("X_missing", 0) > 0:
        st.warning(f"⚠️ {score_data.get('X_missing')} mandatory section(s) missing.")
    if resume_data.total_page_count > 1:
        st.warning(f"⚠️ Resume is {resume_data.total_page_count} pages — keep to 1 page.")
    if score_data.get("buzzwords_found"):
        words = ", ".join(f"'{w}' ×{c}" for w, c in score_data.get("buzzwords_found", {}).items())
        st.warning(f"⚠️ Buzzwords: {words}")
    if score_data.get("S_realization", 100) < 50:
        st.warning("⚠️ Many declared skills not found in project text.")
    if score_data.get("S_complexity", 100) < 65:
        st.warning("⚠️ No Tier 3 project — add Docker, Redis, Kafka, WebSockets, or cloud infra.")
    if score_data.get("S_production", 100) < 50:
        st.warning("⚠️ Low production score — add GitHub + live deployment links.")

    # ── Skills ──
    if resume_data.skills_section_keywords:
        st.markdown("---")
        st.markdown("### 🛠️ Detected Skills")
        st.markdown(" ".join(
            f'<span style="background:#dcfce7;color:#14532d;padding:.2rem .6rem;border-radius:20px;font-size:.82rem;margin:.2rem;display:inline-block;">{s}</span>'
            for s in resume_data.skills_section_keywords
        ), unsafe_allow_html=True)

    if resume_data.domain_classification_vector:
        st.markdown("**Domains:**")
        st.markdown(" ".join(
            f'<span style="background:#bbf7d0;color:#14532d;padding:.2rem .8rem;border-radius:20px;font-size:.82rem;font-weight:600;margin:.2rem;display:inline-block;">{d}</span>'
            for d in set(resume_data.domain_classification_vector)
        ), unsafe_allow_html=True)

#  TAB:  FORMULA STEPS
# ═══════════════════════════════════════════════
elif dashboard == "🔬 Formula Steps":
    st.markdown("### 🔬 Step-by-Step Formula Evaluation")
    st.caption("Every intermediate value shown — verify the math yourself.")

    steps = score_data.get("steps", {})
    C = overrides["constants"]

    def step(num, text):
        st.markdown(f'<div class="step-row"><div class="step-num">{num}</div><div class="step-text">{text}</div></div>', unsafe_allow_html=True)

    # ── 1. Hygiene ──
    with st.expander("🏗️ S_hygiene — Structural Hygiene", expanded=True):
        st.markdown(f'<div class="formula-box">S_hygiene = max(0, 100 − {C["hygiene_page_pen"]}·max(0,P−1) − {C["hygiene_link_pen"]}·L_missing − {C["hygiene_email_pen"]}·E_generic − {C["hygiene_sec_pen"]}·|X_missing|)</div>', unsafe_allow_html=True)
        h = steps.get("hygiene", {})
        step(1, f"Page count P = <span class='step-val'>{h.get('P','-')}</span> → page penalty = <span class='step-val'>{C['hygiene_page_pen']} × max(0, {h.get('P',1)}−1) = {C['hygiene_page_pen'] * max(0, h.get('P',1)-1)}</span>")
        step(2, f"GitHub present: <span class='step-val'>{h.get('has_github','?')}</span> | LinkedIn present: <span class='step-val'>{h.get('has_linkedin','?')}</span> → L_missing = <span class='step-val'>{h.get('L_missing','?')}</span> → penalty = <span class='step-val'>{C['hygiene_link_pen']} × {h.get('L_missing',0)} = {C['hygiene_link_pen'] * h.get('L_missing',0)}</span>")
        step(3, f"Email flag E_generic = <span class='step-val'>{h.get('E_generic','?')}</span> → penalty = <span class='step-val'>{C['hygiene_email_pen'] * h.get('E_generic',0)}</span>")
        step(4, f"Missing sections X_missing = <span class='step-val'>{h.get('X_missing','?')}</span> → penalty = <span class='step-val'>{C['hygiene_sec_pen'] * h.get('X_missing',0)}</span>")
        step(5, f"<b>S_hygiene = max(0, 100 − {C['hygiene_page_pen']*max(0,h.get('P',1)-1)} − {C['hygiene_link_pen']*h.get('L_missing',0)} − {C['hygiene_email_pen']*h.get('E_generic',0)} − {C['hygiene_sec_pen']*h.get('X_missing',0)}) = <span class='step-val'>{h.get('score','?')}</span></b>")

    # ── 2. Realization ──
    with st.expander("🔗 S_realization — Tech-Stack Realization (Complexity-Weighted)"):
        st.markdown('<div class="formula-box">S_realization = ( Σ ln(D_k+1) for k ∈ (declared ∩ applied) )\n               ÷ ( Σ ln(D_k+1) for k ∈ declared + ε ) × 100\n\nD_k: Tier1=2, Tier2=5, Tier3=10</div>', unsafe_allow_html=True)
        r = steps.get("realization", {})
        step(1, f"Skills declared in Skills section: <span class='step-val'>{r.get('declared_count','?')}</span>")
        step(2, f"Skills found in project/experience text: <span class='step-val'>{r.get('applied_count','?')}</span>")
        step(3, f"Intersection (verified): <span class='step-val'>{r.get('intersect_count','?')}</span> skills → {', '.join(r.get('verified',[])[:6]) or 'none'}")
        step(4, f"Unverified skills (declared but not in project text): {', '.join(r.get('unverified',[])[:6]) or 'none'}")
        step(5, f"Σ ln(D_k+1) for intersect = <span class='step-val'>{r.get('sum_intersect','?')}</span>")
        step(6, f"Σ ln(D_k+1) for declared + ε = <span class='step-val'>{r.get('sum_declared','?')}</span>")
        step(7, f"<b>S_realization = ({r.get('sum_intersect','?')} ÷ {r.get('sum_declared','?')}) × 100 = <span class='step-val'>{r.get('score','?')}</span></b>")

    # ── 3. Complexity ──
    with st.expander("⚙️ S_complexity — Project Architectural Complexity (Non-Linear Dominance)"):
        st.markdown(f'<div class="formula-box">S_complexity = min(100, max(C_j) + α·ln(|J|+1))\nα = {C["alpha"]}  |  Tier1={C["tier1_c"]}  Tier2={C["tier2_c"]}  Tier3={C["tier3_c"]}</div>', unsafe_allow_html=True)
        cx = steps.get("complexity", {})
        for title, tier in cx.get("project_tiers", []):
            tcls = "tier3" if tier==C["tier3_c"] else ("tier2" if tier==C["tier2_c"] else "tier1")
            tlabel = "Tier 3" if tier==C["tier3_c"] else ("Tier 2" if tier==C["tier2_c"] else "Tier 1")
            step("→", f"{title} → <span class='{tcls}'>{tlabel} (C_j = {tier})</span>")
        step(1, f"max(C_j) across all projects = <span class='step-val'>{cx.get('max_cj','?')}</span>")
        step(2, f"|J| = {cx.get('J','?')} projects → α·ln(|J|+1) = {cx.get('alpha','?')} × ln({cx.get('J',0)+1}) = <span class='step-val'>{cx.get('log_bonus','?')}</span>")
        step(3, f"<b>S_complexity = min(100, {cx.get('max_cj','?')} + {cx.get('log_bonus','?')}) = <span class='step-val'>{cx.get('score','?')}</span></b>")

    # ── 4. Impact ──
    with st.expander("📈 S_impact — Quantifiable Impact (Log-Dampened Saturation)"):
        st.markdown(f'<div class="formula-box">S_impact = min(100, β · Σ log₁₀(V_b + 1))\nβ = {C["beta"]}\n\nV_b conversions: percentage→raw int, users→count, ms→ms value, 1st place→100</div>', unsafe_allow_html=True)
        im = steps.get("impact", {})
        vals = im.get("values", [])
        if vals:
            for v in vals:
                contrib = round(math.log10(v+1), 4) if v > 0 else 0
                step("→", f"V_b = <span class='step-val'>{v}</span> → log₁₀({v}+1) = <span class='step-val'>{contrib}</span>")
        else:
            step("→", "No quantifiable metric values extracted")
        step(1, f"Σ log₁₀(V_b+1) = <span class='step-val'>{im.get('log_sum','?')}</span>")
        step(2, f"<b>S_impact = min(100, {im.get('beta','?')} × {im.get('log_sum','?')}) = <span class='step-val'>{im.get('score','?')}</span></b>")

    # ── 5. Production ──
    with st.expander("🚀 S_production — Code Inception & Hosting Hygiene"):
        st.markdown('<div class="formula-box">S_production = ( J_live_code + J_live_deployment ) / (2 × |J|) × 100</div>', unsafe_allow_html=True)
        pr = steps.get("production", {})
        step(1, f"Total projects |J| = <span class='step-val'>{pr.get('J','?')}</span>")
        step(2, f"Projects with code repo (GitHub/GitLab) = <span class='step-val'>{pr.get('J_code','?')}</span>")
        step(3, f"Projects with live deployment (Vercel/Netlify/Heroku/AWS) = <span class='step-val'>{pr.get('J_deploy','?')}</span>")
        step(4, f"<b>S_production = ({pr.get('J_code','?')} + {pr.get('J_deploy','?')}) / (2 × {pr.get('J','?')}) × 100 = <span class='step-val'>{pr.get('score','?')}</span></b>")

    # ── 6. Clarity ──
    with st.expander("🔇 S_clarity — Linguistic Noise & Buzzword Entropy"):
        st.markdown(f'<div class="formula-box">S_clarity = max(0, 100 − ω · Σ ln(count(w)+1))\nω = {C["omega"]}</div>', unsafe_allow_html=True)
        cl = steps.get("clarity", {})
        bmap = cl.get("buzzwords", {})
        if bmap:
            for w, cnt in bmap.items():
                contrib = round(cl.get("omega", 15) * math.log(cnt+1), 4)
                step("→", f"'{w}' appears <span class='step-val'>{cnt}×</span> → {cl.get('omega',15)} × ln({cnt}+1) = <span class='step-val'>{contrib}</span>")
        else:
            step("→", "No buzzwords detected ✅")
        step(1, f"Total deduction = <span class='step-val'>{cl.get('deduction','?')}</span>")
        step(2, f"<b>S_clarity = max(0, 100 − {cl.get('deduction','?')}) = <span class='step-val'>{cl.get('score','?')}</span></b>")

    # ── 7. Domain ──
    with st.expander("🎯 S_domain — Knowledge Domain Divergence"):
        st.markdown('<div class="formula-box">S_domain = 100 × (1 − Unique Domains Found / (Total Tech Stacks + ε))</div>', unsafe_allow_html=True)
        dm = steps.get("domain", {})
        step(1, f"Total skills listed = <span class='step-val'>{dm.get('total_skills','?')}</span>")
        step(2, f"Unique domains mapped: {', '.join(dm.get('domains',[])[:6]) or 'none'} → <span class='step-val'>{dm.get('unique_domains','?')}</span>")
        step(3, f"<b>S_domain = 100 × (1 − {dm.get('unique_domains','?')} / {dm.get('total_skills','?')}) = <span class='step-val'>{dm.get('score','?')}</span></b>")

    # ── 8. Velocity ──
    with st.expander("⏱️ S_velocity — Chronological Velocity"):
        st.markdown('<div class="formula-box">S_velocity = min(100, Σ T_e × μ_e)\nμ_e: Internship=15  Freelance/Tech Lead=10  Member=3</div>', unsafe_allow_html=True)
        vl = steps.get("velocity", {})
        for e in vl.get("entries", []):
            mu = vl.get("role_weights", {}).get(e.get("type","member"), 3)
            contrib = e.get("months", 0) * mu
            step("→", f"{e.get('role','?')} ({e.get('type','?')}) — <span class='step-val'>{e.get('months',0)} months × μ={mu} = {contrib}</span>")
        step(1, f"Σ (T_e × μ_e) = <span class='step-val'>{vl.get('raw_sum','?')}</span>")
        step(2, f"<b>S_velocity = min(100, {vl.get('raw_sum','?')}) = <span class='step-val'>{vl.get('score','?')}</span></b>")

    # ── Final weighted sum ──
    st.markdown("---")
    st.markdown("### 🧮 Final Score — Dot Product")
    st.markdown(f'<div class="formula-box">S_final = Σ (w_i × S_i)\n\n' +
        "\n".join(f"  {label:28s} {W.get(wk,0):.2f} × {score_data.get(sk,0):6.2f} = {round(W.get(wk,0)*score_data.get(sk,0),3)}"
                  for sk, label, wk, _ in COMP_KEYS) +
        f"\n{'─'*50}\n  {'FINAL':28s}              = {score_data.get('final_score','?')}"
        + '</div>', unsafe_allow_html=True)

#  TAB: CONTROLS
# ═══════════════════════════════════════════════
elif dashboard == "🎛️ Controls":
    st.markdown("### 🎛️ Weight & Constant Controls")
    st.caption("Adjust any value and click **Apply & Recompute** to instantly see the effect on the score.")

    ov = get_overrides()

    st.markdown("#### ⚖️ Dynamic Weight Matrix")
    st.caption("Each row must sum to 1.00. Adjust per B.Tech year.")

    WLABELS = [
        ("hyg",  "Structural Hygiene"),
        ("real", "Tech Realization"),
        ("comp", "Project Complexity"),
        ("imp",  "Metric Impact"),
        ("prod", "Production Ready"),
        ("clar", "Clarity / Buzzwords"),
        ("dom",  "Domain Focus"),
        ("vel",  "Activity Velocity"),
    ]

    tabs = st.tabs(["Year 2", "Year 3", "Year 4"])
    new_weights = {yr: dict(w) for yr, w in ov["weights"].items()}

    for ti, yr in enumerate([2, 3, 4]):
        with tabs[ti]:
            cols = st.columns(4)
            for i, (wk, wlabel) in enumerate(WLABELS):
                with cols[i % 4]:
                    new_weights[yr][wk] = st.number_input(
                        wlabel, min_value=0.0, max_value=1.0, step=0.01,
                        value=float(ov["weights"][yr].get(wk, 0.0)),
                        key=f"w_{yr}_{wk}", format="%.2f"
                    )
            wsum = sum(new_weights[yr].values())
            color = "green" if abs(wsum - 1.0) < 0.001 else "red"
            st.markdown(f"**Weight sum: :{color}[{wsum:.3f}]** {'✅' if abs(wsum-1.0)<0.001 else '⚠️ must equal 1.00'}")

    st.markdown("---")
    st.markdown("#### 🔢 Formula Constants")

    cc = st.columns(4)
    new_C = dict(ov["constants"])

    with cc[0]:
        st.markdown('<p class="ctrl-header">S_complexity</p>', unsafe_allow_html=True)
        new_C["alpha"]   = st.number_input("α (volume bonus)",  value=float(ov["constants"]["alpha"]),  step=0.5,  key="c_alpha")
        new_C["tier1_c"] = st.number_input("Tier 1 score",      value=int(ov["constants"]["tier1_c"]),  step=5,    key="c_t1")
        new_C["tier2_c"] = st.number_input("Tier 2 score",      value=int(ov["constants"]["tier2_c"]),  step=5,    key="c_t2")
        new_C["tier3_c"] = st.number_input("Tier 3 score",      value=int(ov["constants"]["tier3_c"]),  step=5,    key="c_t3")

    with cc[1]:
        st.markdown('<p class="ctrl-header">S_impact</p>', unsafe_allow_html=True)
        new_C["beta"]    = st.number_input("β (saturation mult)", value=float(ov["constants"]["beta"]), step=0.5,  key="c_beta")

    with cc[2]:
        st.markdown('<p class="ctrl-header">S_clarity</p>', unsafe_allow_html=True)
        new_C["omega"]   = st.number_input("ω (buzzword pen)",  value=float(ov["constants"]["omega"]),  step=1.0,  key="c_omega")

    with cc[3]:
        st.markdown('<p class="ctrl-header">S_hygiene Penalties</p>', unsafe_allow_html=True)
        new_C["hygiene_page_pen"]  = st.number_input("Per extra page",   value=float(ov["constants"]["hygiene_page_pen"]),  step=5.0, key="c_pp")
        new_C["hygiene_link_pen"]  = st.number_input("Per missing link", value=float(ov["constants"]["hygiene_link_pen"]),  step=5.0, key="c_lp")
        new_C["hygiene_email_pen"] = st.number_input("Email flag",       value=float(ov["constants"]["hygiene_email_pen"]), step=5.0, key="c_ep")
        new_C["hygiene_sec_pen"]   = st.number_input("Per missing section", value=float(ov["constants"]["hygiene_sec_pen"]),step=5.0, key="c_sp")

    st.markdown("---")
    col_apply, col_reset = st.columns([1, 1])

    with col_apply:
        if st.button("✅ Apply & Recompute", use_container_width=True, type="primary"):
            st.session_state["overrides"] = {"weights": new_weights, "constants": new_C}
            if "resume_data" in st.session_state:
                new_score = compute_score_custom(st.session_state["resume_data"],
                                                  st.session_state["overrides"])
                st.session_state["score_data"] = new_score
            st.success("Weights applied — switch to 📊 Score or 🔬 Formula Steps to see updated results.")
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset to PDF Defaults", use_container_width=True):
            st.session_state["overrides"] = {
                "weights": {yr: dict(w) for yr, w in DEFAULT_WEIGHTS.items()},
                "constants": dict(DEFAULT_CONSTANTS),
            }
            if "resume_data" in st.session_state:
                new_score = compute_score_custom(st.session_state["resume_data"],
                                                  st.session_state["overrides"])
                st.session_state["score_data"] = new_score
            st.success("Reset to original PDF values.")
            st.rerun()

    # ── Live preview ──
    st.markdown("---")
    st.markdown("#### 👁️ Current vs Default Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current overrides (Year 3):**")
        for wk, wlabel in WLABELS:
            cur = ov["weights"][3].get(wk, 0)
            dfl = DEFAULT_WEIGHTS[3].get(wk, 0)
            diff = cur - dfl
            marker = f" ({'↑' if diff>0 else '↓'}{abs(diff):.2f})" if abs(diff) > 0.001 else ""
            st.markdown(f"- {wlabel}: **{cur:.2f}**{marker}")
    with c2:
        st.markdown("**Current constants:**")
        for ck, cv in ov["constants"].items():
            dfl = DEFAULT_CONSTANTS.get(ck, cv)
            diff = cv - dfl
            marker = f" ({'↑' if diff>0 else '↓'}{abs(diff):.2f})" if abs(diff) > 0.001 else ""
            st.markdown(f"- {ck}: **{cv}**{marker}")

#  TAB: RAW DATA
# ═══════════════════════════════════════════════
elif dashboard == "🗂️ Raw Data":
    st.markdown("### 🗂️ LLM-Extracted Resume Data")
    st.caption("Raw JSON output from the extraction step — fed directly into the math engine.")

    out = {
        "candidate": {"name": resume_data.candidate_name, "btech_year": resume_data.btech_year},
        "hygiene": {
            "total_page_count": resume_data.total_page_count,
            "extracted_links": resume_data.extracted_links_array,
            "email": resume_data.raw_email_string,
            "section_headers": resume_data.detected_section_headers,
        },
        "skills": {
            "section_keywords": resume_data.skills_section_keywords,
            "count": len(resume_data.skills_section_keywords),
            "domain_vector": resume_data.domain_classification_vector,
        },
        "projects": {
            "count": resume_data.project_count,
            "titles": resume_data.project_titles,
            "tech_per_project": resume_data.project_tech_keywords,
            "arch_flags": resume_data.architectural_regex_flags,
            "code_repos": resume_data.code_repository_urls,
            "live_deployments": resume_data.deployment_live_urls,
        },
        "impact": {
            "total_bullets": resume_data.total_bullet_points_count,
            "metric_bullets": resume_data.metric_regex_match_count,
            "numeric_values": resume_data.regex_extracted_numeric_values,
        },
        "clarity": {"buzzword_frequency_map": resume_data.buzzword_frequency_map},
        "experience": {"timeline": resume_data.experience_timeline_intervals},
        "scores": {
            "final": score_data.get("final_score", 0),
            "year_weights": score_data.get("btech_year", 3),
            **{k: score_data.get(k, 0) for k in
               ["S_hygiene","S_realization","S_complexity","S_impact",
                "S_production","S_clarity","S_domain","S_velocity"]},
        },
    }
    json_str = json.dumps(out, indent=2)
    st.code(json_str, language="json")
    st.download_button("⬇️ Download JSON", data=json_str,
        file_name=f"{resume_data.candidate_name.replace(' ','_')}_data.json",
        mime="application/json", use_container_width=True)
