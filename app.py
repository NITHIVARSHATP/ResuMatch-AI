import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ResuMatch AI - Candidate Screening Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
from src.logger import logger
from src.database import (
    init_db,
    save_candidate,
    load_candidates,
    delete_candidate,
    is_candidate_exists,
    wipe_db
)
from src.parser import extract_text, parse_resume_text
from src.matcher import analyze_candidate_match
from src.rag_engine import build_vector_store, answer_candidate_query

# Load Custom CSS Style
def load_css():
    css_path = os.path.join("assets", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning("Custom CSS file not found at assets/custom.css")

load_css()

# Initialize Database
init_db()

# Initialize Session State Variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("GEMINI_API_KEY", "")

if "matches" not in st.session_state:
    st.session_state["matches"] = {}

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Helper to validate Gemini API Key
def is_api_key_valid(api_key):
    if not api_key or len(api_key) < 10:
        return False
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )
        # Simple test invocation (fast & lightweight)
        llm.invoke("Hi")
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

# Initialize vector store on startup if API key is valid and candidates exist
if st.session_state["api_key"] and st.session_state["vector_store"] is None:
    candidates = load_candidates()
    if candidates:
        try:
            # We don't want to block UI startup if it fails
            st.session_state["vector_store"] = build_vector_store(candidates, api_key=st.session_state["api_key"])
        except Exception as e:
            logger.error(f"Startup vector store creation failed: {e}")

# Sidebar UI
st.sidebar.markdown('<h1 class="text-gradient" style="font-size: 2.2rem; margin-bottom: 20px;">ResuMatch AI</h1>', unsafe_allow_html=True)

# 1. API Validation Box
st.sidebar.subheader("API Configuration")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=st.session_state["api_key"],
    type="password",
    help="Enter your Gemini API Key. If empty, the app will try to read GEMINI_API_KEY from the environment."
)

if api_key_input != st.session_state["api_key"]:
    st.session_state["api_key"] = api_key_input
    # Clear vector store cache to force rebuild with new key
    st.session_state["vector_store"] = None

# Show API status badge
if st.session_state["api_key"]:
    # Validate the key cached in state
    if "api_key_status" not in st.session_state or st.session_state.get("last_checked_key") != st.session_state["api_key"]:
        with st.spinner("Validating API Key..."):
            valid = is_api_key_valid(st.session_state["api_key"])

            st.session_state["api_key_status"] = valid
            st.session_state["last_checked_key"] = st.session_state["api_key"]
            
    if st.session_state.get("api_key_status"):
        st.sidebar.success("API Key Active & Valid")
    else:
        st.sidebar.error("Invalid Gemini API Key")
else:
    st.sidebar.warning("Missing API Key. Provide it above.")

st.sidebar.markdown("---")

# 2. Match Score Filter
st.sidebar.subheader("Leaderboard Settings")
match_threshold = st.sidebar.slider(
    "Shortlist Score Threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="Candidates scoring at or above this threshold will be categorized as shortlisted."
)

st.sidebar.markdown("---")

# 3. Database Manager
st.sidebar.subheader("Database Manager")
all_candidates = load_candidates()

if all_candidates:
    candidate_options = {c["filename"]: c for c in all_candidates}
    selected_del_filename = st.sidebar.selectbox(
        "Select Resume to Delete",
        options=list(candidate_options.keys()),
        index=0
    )
    
    if st.sidebar.button("🗑️ Delete Selected Resume", use_container_width=True):
        candidate_to_del = candidate_options[selected_del_filename]
        success = delete_candidate(selected_del_filename)
        if success:
            st.sidebar.success(f"Deleted {candidate_to_del['name']}'s resume")
            # Clear matches and trigger rebuild of vector store
            if selected_del_filename in st.session_state["matches"]:
                del st.session_state["matches"][selected_del_filename]
            st.session_state["vector_store"] = None
            st.rerun()
        else:
            st.sidebar.error("Failed to delete candidate.")
else:
    st.sidebar.info("No candidates in the database.")

if st.sidebar.button("🚨 Wipe DB & Clear Cache", use_container_width=True, type="secondary"):
    if wipe_db():
        st.session_state["matches"] = {}
        st.session_state["vector_store"] = None
        st.session_state["chat_history"] = []
        st.sidebar.success("Database and caches successfully cleared!")
        st.rerun()
    else:
        st.sidebar.error("Failed to wipe database.")

# Main Application Dashboard
st.markdown('<h1 class="title-gradient">ResuMatch AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;">AI-Powered Resume Screening, Applicant Tracking, and Candidate Evaluation System.</p>', unsafe_allow_html=True)

# Tabs Definition
tab1, tab2, tab3 = st.tabs([
    "🔍 Screening Dashboard", 
    "👤 Candidate Profiler", 
    "💬 Q&A Assistant"
])

# ==============================================================================
# TAB 1: SCREENING DASHBOARD
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 1.1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### 📋 Job Description & Upload', unsafe_allow_html=True)
        
        # JD Input
        jd_text = st.text_area(
            "Paste Job Description (JD)",
            height=200,
            placeholder="Paste the job description or requirement list here...",
            key="jd_input"
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="uploader"
        )
        
        # Process Button
        process_btn = st.button("🚀 Process Screening", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if process_btn:
            if not st.session_state["api_key"]:
                st.error("Please configure your Google Gemini API Key in the sidebar first!")
            elif not jd_text.strip():
                st.error("Please enter a Job Description to match candidates against!")
            elif not uploaded_files and not all_candidates:
                st.error("Please upload at least one resume to process!")
            else:
                try:
                    # Progress indicators
                    progress_text = st.empty()
                    progress_bar = st.progress(0.0)
                    
                    # 1. Parse and save new candidates
                    new_candidates_count = len(uploaded_files)
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        filename = uploaded_file.name
                        progress_text.markdown(f"**Step 1/3: Extracting & Parsing {filename}...**")
                        progress_bar.progress(float(idx) / float(new_candidates_count * 3))
                        
                        file_bytes = uploaded_file.getvalue()
                        
                        # Only parse and save if not already processed in SQLite
                        if not is_candidate_exists(filename):
                            try:
                                # Extract raw text
                                raw_text = extract_text(file_bytes, filename)
                                # Parse schema using LLM
                                parsed_details = parse_resume_text(raw_text, api_key=st.session_state["api_key"])
                                # Save candidate
                                save_candidate(
                                    filename=filename,
                                    name=parsed_details.name,
                                    raw_text=raw_text,
                                    parsed_details=parsed_details.model_dump(),
                                    file_bytes=file_bytes
                                )
                            except Exception as parse_err:
                                logger.error(f"Failed to process {filename}: {parse_err}")
                                st.warning(f"Failed to parse resume '{filename}': {parse_err}")
                    
                    # Refresh candidates list
                    all_candidates = load_candidates()
                    
                    # 2. Build or Rebuild Vector Store
                    progress_text.markdown("**Step 2/3: Rebuilding Semantic Vector Store...**")
                    progress_bar.progress(0.66)
                    st.session_state["vector_store"] = build_vector_store(
                        all_candidates, 
                        api_key=st.session_state["api_key"]
                    )
                    
                    # 3. Match Evaluation
                    progress_text.markdown("**Step 3/3: Running Match Fit Evaluations...**")
                    total_candidates = len(all_candidates)
                    
                    for idx, cand in enumerate(all_candidates):
                        filename = cand["filename"]
                        cand_name = cand["name"]
                        raw_text = cand["raw_text"]
                        
                        progress_text.markdown(f"**Step 3/3: Matching {cand_name}...**")
                        progress_bar.progress(0.66 + (0.33 * float(idx) / float(total_candidates)))
                        
                        try:
                            # Run Match Evaluation
                            match_analysis = analyze_candidate_match(
                                resume_text=raw_text,
                                jd_text=jd_text,
                                candidate_name=cand_name,
                                api_key=st.session_state["api_key"]
                            )
                            st.session_state["matches"][filename] = match_analysis
                        except Exception as match_err:
                            logger.error(f"Failed matching for {cand_name}: {match_err}")
                            st.warning(f"Could not complete matching for candidate '{cand_name}': {match_err}")
                            
                    progress_text.empty()
                    progress_bar.empty()
                    st.success("Screening process finished successfully! Review results on the right.")
                    st.rerun()
                except Exception as proc_e:
                    logger.error(f"Screening execution error: {proc_e}")
                    st.error(f"Screening Execution Failed: {proc_e}")
    
    with col2:
        if st.session_state["matches"]:
            st.markdown("### 📊 Screening Overview")
            
            # Compute Metrics
            matches_list = []
            for filename, match in st.session_state["matches"].items():
                matches_list.append({
                    "filename": filename,
                    "name": match.candidate_name,
                    "score": match.match_percentage,
                    "overall_fit": match.overall_fit_explanation
                })
            
            df = pd.DataFrame(matches_list)
            df = df.sort_values(by="score", ascending=False)
            
            total_resumes = len(df)
            shortlisted_df = df[df["score"] >= match_threshold]
            total_shortlisted = len(shortlisted_df)
            avg_score = int(df["score"].mean()) if not df.empty else 0
            
            # Displays Metric Cards
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-val">{total_resumes}</div>
                    <div class="metric-lbl">Total Screened</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val" style="color: #10b981;">{total_shortlisted}</div>
                    <div class="metric-lbl">Shortlisted (Score &ge; {match_threshold}%)</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val" style="color: #818cf8;">{avg_score}%</div>
                    <div class="metric-lbl">Average Match</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Leaderboard with glassmorphism cards
            st.markdown("#### 🏆 Leaderboard")
            for index, row in df.iterrows():
                score = row["score"]
                # Determine score style class
                if score >= 80:
                    badge_class = "badge-high"
                elif score >= 50:
                    badge_class = "badge-medium"
                else:
                    badge_class = "badge-low"
                    
                st.markdown(f"""
                <div class="glass-card" style="padding: 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 1.25rem; font-weight: 700; color: #818cf8; min-width: 24px;">#{index + 1}</span>
                        <div>
                            <div style="font-weight: 600; font-size: 1.05rem; color: #ffffff;">{row['name']}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">{row['filename']}</div>
                        </div>
                    </div>
                    <span class="score-badge {badge_class}">{score}% Match</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Plotly Horizontal Bar Chart
            st.markdown("#### 📈 Candidate Match Scores")
            fig = px.bar(
                df,
                x="score",
                y="name",
                orientation="h",
                labels={"score": "Match Percentage (%)", "name": "Candidate"},
                color="score",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                range_x=[0, 100]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f1f5f9",
                height=300 + (len(df) * 20),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis={'categoryorder':'total ascending'}
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No candidates screened yet. Enter a Job Description, upload resumes, and click 'Process Screening' to begin analysis.")

# ==============================================================================
# TAB 2: CANDIDATE PROFILER
# ==============================================================================
with tab2:
    if not all_candidates:
        st.info("No candidates available in the database. Please upload and screen candidates in the 'Screening Dashboard' first.")
    else:
        # Create a dropdown selector to choose candidates
        candidate_names_mapping = {c["filename"]: f"{c['name']} ({c['filename']})" for c in all_candidates}
        selected_file = st.selectbox(
            "🔍 Select Candidate to Profile",
            options=list(candidate_names_mapping.keys()),
            format_func=lambda x: candidate_names_mapping[x]
        )
        
        # Load active candidate object
        active_cand = next(c for c in all_candidates if c["filename"] == selected_file)
        details = active_cand["parsed_details"]
        
        # Attempt to load their match evaluation
        match_info = st.session_state["matches"].get(selected_file)
        
        # Name and Score Badge Title row
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1:
            st.markdown(f'<h2 style="margin-bottom:5px; color:#ffffff;">{details.get("name", "Unknown Candidate")}</h2>', unsafe_allow_html=True)
            
            # Contact details row
            email = details.get("email", "")
            phone = details.get("phone", "")
            linkedin = details.get("linkedin", "")
            
            contact_html = []
            if email:
                contact_html.append(f"📧 **Email:** {email}")
            if phone:
                contact_html.append(f"📞 **Phone:** {phone}")
            if linkedin:
                # Make sure it has http prefix for clickable URL
                link_url = linkedin if linkedin.startswith("http") else f"https://{linkedin}"
                contact_html.append(f'🔗 <a href="{link_url}" target="_blank" style="color:#818cf8; text-decoration:none;"><b>LinkedIn Profile</b></a>')
                
            if contact_html:
                st.markdown(" | ".join(contact_html), unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#94a3b8; font-style:italic;'>No contact information extracted.</p>", unsafe_allow_html=True)
                
        with c_col2:
            if match_info:
                score = match_info.match_percentage
                if score >= 80:
                    badge_class = "badge-high"
                elif score >= 50:
                    badge_class = "badge-medium"
                else:
                    badge_class = "badge-low"
                    
                st.markdown(f"""
                <div style="text-align: right; margin-top: 10px;">
                    <span class="score-badge {badge_class}" style="font-size: 1.35rem; padding: 10px 24px;">{score}% Match Score</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: right; margin-top: 10px;">
                    <span class="score-badge badge-low" style="font-style: italic; font-size: 1.1rem; padding: 8px 18px;">Not Screened Against Current JD</span>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")
        
        # Strengths & Gaps Side-by-Side (only if screened)
        if match_info:
            col_fit1, col_fit2 = st.columns(2)
            
            with col_fit1:
                st.markdown('<div class="glass-card" style="border-color: rgba(16, 185, 129, 0.2); min-height: 250px;">', unsafe_allow_html=True)
                st.markdown('<h4 style="color:#10b981; margin-top:0;">💪 Key Alignments & Strengths</h4>', unsafe_allow_html=True)
                
                strengths = match_info.key_alignment_points
                if strengths:
                    for pt in strengths:
                        st.markdown(f'<div class="fit-list-item"><span class="fit-list-icon strength-icon">✓</span><span>{pt}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:#94a3b8;'>No explicit strengths highlighted.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_fit2:
                st.markdown('<div class="glass-card" style="border-color: rgba(239, 68, 68, 0.2); min-height: 250px;">', unsafe_allow_html=True)
                st.markdown('<h4 style="color:#ef4444; margin-top:0;">⚠️ Missing Requirements & Gaps</h4>', unsafe_allow_html=True)
                
                gaps = match_info.missing_requirements
                if gaps:
                    for pt in gaps:
                        st.markdown(f'<div class="fit-list-item"><span class="fit-list-icon gap-icon">✗</span><span>{pt}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:#10b981;'><span class='fit-list-icon strength-icon'>✓</span>No critical missing requirements identified!</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Overall Fit Explanation Card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4>🧐 Overall Fit Explanation</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="line-height:1.6; color:#e2e8f0;">{match_info.overall_fit_explanation}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Extracted Details Tabs
        st.markdown("### 📝 Extracted Resume Details")
        detail_tab1, detail_tab2, detail_tab3 = st.tabs(["💡 Skills & Certifications", "💼 Work History", "🎓 Education"])
        
        with detail_tab1:
            skills = details.get("skills", [])
            if skills:
                st.markdown('<div style="padding: 10px 0;">', unsafe_allow_html=True)
                for skill in skills:
                    st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No skills extracted from this candidate's resume.")
                
        with detail_tab2:
            experience = details.get("experience", [])
            if experience:
                for exp in experience:
                    company = exp.get("company", "Unknown Company")
                    job_title = exp.get("job_title", "Unknown Role")
                    duration = exp.get("duration", "N/A")
                    desc = exp.get("description", "")
                    
                    st.markdown(f"""
                    <div class="experience-card">
                        <div class="experience-title">{job_title}</div>
                        <div class="experience-company">{company}</div>
                        <div class="experience-duration">⏳ {duration}</div>
                        <p style="color:#cbd5e1; font-size:0.95rem; margin-top:8px; line-height:1.5;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No work experience details extracted from this candidate's resume.")
                
        with detail_tab3:
            education = details.get("education", [])
            if education:
                for edu in education:
                    degree = edu.get("degree", "Degree / Certification")
                    inst = edu.get("institution", "Institution / University")
                    year = edu.get("graduation_year", "")
                    
                    year_str = f" | Class of {year}" if year else ""
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                        <div style="font-weight:600; font-size:1.05rem; color:#ffffff;">{degree}</div>
                        <div style="color:#a5b4fc; font-size:0.9rem; margin-top:2px;">🏫 {inst}{year_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No education history extracted from this candidate's resume.")

# ==============================================================================
# TAB 3: Q&A ASSISTANT
# ==============================================================================
with tab3:
    st.markdown("### 💬 Resume RAG Q&A Assistant")
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem;'>Ask natural language questions about candidate qualifications. The assistant will answer using facts extracted directly from the resumes.</p>", unsafe_allow_html=True)
    
    # Check if vector store is initialized
    if st.session_state["vector_store"] is None:
        st.info("The vector store is currently empty. Please upload resumes and click 'Process Screening' in the screening dashboard to populate and build the vector database.")
    else:
        # Chat Context scope
        qa_candidates = ["All Candidates"] + [c["name"] for c in all_candidates]
        chat_context = st.selectbox(
            "🎯 Scoped Query Context",
            options=qa_candidates,
            index=0,
            help="Select 'All Candidates' to search across the entire resume repository, or select a specific candidate to focus the retrieval strictly on their resume."
        )
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state["chat_history"]:
                role_class = "chat-user" if message["role"] == "user" else "chat-ai"
                role_label = "👤 You" if message["role"] == "user" else "🤖 Recruiter AI"
                
                # Check for context tag if present
                context_tag = f" <span style='font-size:0.75rem; color:#818cf8; border:1px solid rgba(99, 102, 241, 0.4); padding:2px 6px; border-radius:4px; margin-left:8px;'>{message.get('context', '')}</span>" if 'context' in message else ""
                
                st.markdown(f"""
                <div class="chat-bubble {role_class}">
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 5px; font-weight:600;">{role_label}{context_tag}</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Query input field
        query_text = st.text_input("Ask a question about the candidate(s):", placeholder="e.g. List candidates with experience in Kubernetes, or Tell me about John's Python background...", key="qa_input_text")
        
        col_c1, col_c2 = st.columns([1, 5])
        with col_c1:
            send_btn = st.button("Send ➔", type="primary", use_container_width=True)
        with col_c2:
            clear_btn = st.button("Clear Chat History", use_container_width=True)
            
        if clear_btn:
            st.session_state["chat_history"] = []
            st.rerun()
            
        if send_btn and query_text.strip():
            if not st.session_state["api_key"]:
                st.error("Google Gemini API Key is missing. Please configure it in the sidebar.")
            else:
                # Add user query to conversation history
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": query_text,
                    "context": chat_context
                })
                
                # Retrieve answer from RAG
                with st.spinner("Synthesizing answer from candidate resumes..."):
                    answer = answer_candidate_query(
                        vector_store=st.session_state["vector_store"],
                        query=query_text,
                        candidate_name=chat_context,
                        api_key=st.session_state["api_key"]
                    )
                    
                # Add AI response to conversation history
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()
