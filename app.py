import streamlit as st
import pandas as pd
import time
import re
import json
from utils import extract_text_from_pdf, compute_similarity_and_skills, compare_two_resumes

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI Resume Screener | Smart Hiring",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS WITH ANIMATIONS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #1e1e2f);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .title-animation {
        text-align: center; font-size: 3.5rem; font-weight: 800;
        background: linear-gradient(135deg, #f0f9ff, #38bdf8, #818cf8);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: fadeInUp 0.8s ease-out; margin-bottom: 0.5rem;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Fixed card style – no empty boxes */
    .custom-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 18px 24px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 15px;
    }
    .custom-card h3 {
        margin: 0;
        color: white;
    }
    .stTextArea textarea, .stFileUploader > div {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid #334155 !important;
        border-radius: 16px !important;
        transition: all 0.3s ease;
        color: white !important;
    }
    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.3) !important;
        transform: scale(1.01);
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border: none;
        border-radius: 40px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        color: white;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(59,130,246,0.5);
        background: linear-gradient(90deg, #2563eb, #7c3aed);
    }
    .stDataFrame {
        background: rgba(15,23,42,0.5);
        border-radius: 20px;
        overflow: hidden;
        backdrop-filter: blur(4px);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #f97316);
        border-radius: 20px;
    }
    [data-testid="stSidebar"] {
        background: rgba(15,23,42,0.7);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(56,189,248,0.2);
    }
    .metric-card {
        background: rgba(0,0,0,0.3);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56,189,248,0.3);
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #38bdf8;
        border-radius: 10px;
    }
    /* Hide default Streamlit footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    # Logo (replace with your own if needed)
    st.markdown('<div style="text-align: center; font-size: 3rem;">🤖</div>', unsafe_allow_html=True)
    st.markdown("### ✨ AI Screening Suite")
    st.markdown("---")
    st.markdown("""
    **Smart Features**  
    🧠 NLP-based ranking  
    📄 PDF text extraction  
    ⚡ Real-time similarity  
    🎯 TF-IDF + Cosine  
    ---
    **How it works**  
    1. Paste job description  
    2. Upload PDF resumes  
    3. Click **Screen Resumes**  
    4. Get ranked matches  
    """)
    st.markdown("---")
    
    weight = st.slider("⚖️ Similarity vs Skill Match weight", 0.0, 1.0, 0.5, 0.05,
                       help="0 = only Skill Match, 1 = only Similarity")
    st.markdown("---")
    
    st.markdown("**➕ Add Custom Skills** (comma‑separated)")
    custom_skills_input = st.text_area("", placeholder="e.g., Kotlin, Figma, AWS Lambda", height=68, label_visibility="collapsed")
    custom_skills = [s.strip().lower() for s in custom_skills_input.split(",") if s.strip()]
    if custom_skills:
        st.success(f"{len(custom_skills)} custom skill(s) added")
    st.markdown("---")

# ========== MAIN UI ==========
st.markdown('<div class="title-animation">📄 AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">Match, analyze skills, and rank candidates instantly</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # Fixed card – heading inside a single markdown, text_area placed after
    st.markdown("""
    <div class="custom-card">
        <h3>📝 Job Description</h3>
    </div>
    """, unsafe_allow_html=True)
    jd_text = st.text_area(
        "",
        height=250,
        placeholder="Paste job description here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
    <div class="custom-card">
        <h3>📎 Upload Resumes (PDF)</h3>
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    screen_btn = st.button("🚀 SCREEN RESUMES", use_container_width=True)

# ========== PROCESSING ==========
if screen_btn:
    if not jd_text.strip():
        st.error("❌ Please paste a job description.")
    elif not uploaded_files:
        st.error("❌ Please upload at least one PDF resume.")
    else:
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            resumes_data = []
            failed = []
            
            for idx, pdf in enumerate(uploaded_files):
                status_text.markdown(f"**📄 Extracting:** {pdf.name}")
                raw = extract_text_from_pdf(pdf)
                if not raw.strip():
                    failed.append(pdf.name)
                    continue
                resumes_data.append((pdf.name, raw))
                progress_bar.progress((idx+1)/len(uploaded_files))
            
            if failed:
                st.warning(f"⚠️ Could not extract from: {', '.join(failed)}")
            if not resumes_data:
                st.error("No readable resumes. Aborting.")
                st.stop()
            
            status_text.markdown("**🧠 Computing similarity & skills...**")
            results_df = compute_similarity_and_skills(jd_text, resumes_data, custom_skills=custom_skills)
            progress_bar.empty()
            status_text.empty()
            
            # Combined score
            results_df["Combined Score"] = (weight * results_df["Similarity Score"] * 100) + ((1-weight) * results_df["Skill Match %"])
            results_df["Combined Score"] = results_df["Combined Score"] / 100
            results_df = results_df.sort_values(by="Combined Score", ascending=False)
            
            # Show all resumes (no threshold)
            filtered_df = results_df.copy()
            
            # ========== METRICS ==========
            st.markdown("---")
            st.markdown('<h2 style="text-align: center;">🏆 Screening Results</h2>', unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f'<div class="metric-card"><h3>📄 Resumes</h3><div class="value">{len(resumes_data)}</div></div>', unsafe_allow_html=True)
            with col_m2:
                best_combined = filtered_df.iloc[0]["Combined Score"]
                st.markdown(f'<div class="metric-card"><h3>🏅 Best Combined</h3><div class="value">{best_combined:.2%}</div></div>', unsafe_allow_html=True)
            with col_m3:
                avg_similarity = filtered_df["Similarity Score"].mean()
                st.markdown(f'<div class="metric-card"><h3>📊 Avg Similarity</h3><div class="value">{avg_similarity:.2%}</div></div>', unsafe_allow_html=True)
            with col_m4:
                best_skill = filtered_df.iloc[0]["Skill Match %"]
                st.markdown(f'<div class="metric-card"><h3>🎯 Top Skill Match</h3><div class="value">{best_skill:.1f}%</div></div>', unsafe_allow_html=True)
            
            # ========== RESULTS TABLE ==========
            st.markdown("### 📋 Ranked Candidates (Combined Score)")
            column_config = {
                "Resume": "📄 Resume",
                "Similarity Score": st.column_config.ProgressColumn("🔗 Similarity", format="%.2f", min_value=0, max_value=1),
                "Skill Match %": st.column_config.ProgressColumn("🎯 Skill Match", format="%.1f%%", min_value=0, max_value=100),
                "Combined Score": st.column_config.ProgressColumn("⭐ Combined", format="%.2f", min_value=0, max_value=1),
                "Top Resume Skills": "🏆 Top Skills",
                "Matched Skills": "✅ Matched",
                "Missing Skills": "❌ Missing"
            }
            display_cols = ["Resume", "Similarity Score", "Skill Match %", "Combined Score", "Top Resume Skills", "Matched Skills", "Missing Skills"]
            st.dataframe(filtered_df[display_cols], use_container_width=True, column_config=column_config)
            
            # ========== EXPORT CSV & JSON ==========
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Results as CSV", csv, "resume_screening.csv", "text/csv", use_container_width=True)
            
            detailed_report = {
                "job_description": jd_text,
                "settings": {"similarity_weight": weight, "custom_skills": custom_skills},
                "results": filtered_df.to_dict(orient="records")
            }
            json_str = json.dumps(detailed_report, indent=2)
            st.download_button("📑 Export Detailed JSON Report", json_str, "full_report.json", "application/json", use_container_width=True)
            
            # ========== BAR CHART ==========
            st.markdown("#### 📊 Combined Score Distribution")
            chart_data = filtered_df.set_index("Resume")["Combined Score"]
            st.bar_chart(chart_data, height=400, use_container_width=True)
            
            # ========== RESUME COMPARISON ==========
            
            # ========== CANDIDATE DEEP DIVE ==========
            st.markdown("### 🔍 Candidate Deep Dive")
            for _, row in filtered_df.iterrows():
                with st.expander(f"📄 {row['Resume']} - Combined: {row['Combined Score']:.2%} | Skill Match: {row['Skill Match %']:.1f}%"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**🏆 Top Skills**")
                        st.success(row['Top Resume Skills'])
                        st.markdown("**✅ Matched Skills**")
                        st.success(row['Matched Skills'])
                        st.markdown("**❌ Missing Skills**")
                        st.warning(row['Missing Skills'])
                        if row['Missing Skills'] != "None":
                            rec = f"Consider adding: {', '.join(row['Missing Skills'].split(', ')[:5])}"
                            st.info(f"💡 **Recommendation:** {rec}")
                        # Download original PDF
                        pdf_bytes = next(p for p in uploaded_files if p.name == row['Resume'])
                        st.download_button("📄 Download PDF", data=pdf_bytes, file_name=row['Resume'], mime="application/pdf")
                    with col_b:
                        resume_text = next(t for n, t in resumes_data if n == row['Resume'])
                        preview = resume_text[:1500]
                        if len(resume_text) > 1500:
                            last_space = preview.rfind(' ')
                            if last_space > 0:
                                preview = preview[:last_space] + "..."
                        matched_skills = row['Matched Skills'].split(", ")
                        highlighted = preview
                        for skill in matched_skills:
                            if skill != "None":
                                pattern = re.compile(r'(\b' + re.escape(skill) + r'\b)', re.IGNORECASE)
                                highlighted = pattern.sub(r'<mark style="background:#38bdf8; color:#0f172a; padding:0 2px; border-radius:4px;">\1</mark>', highlighted)
                        st.markdown("**📄 Resume Preview**")
                        st.markdown(f'<div style="background:#0f172a; padding:1rem; border-radius:16px; font-family: monospace; white-space: pre-wrap; max-height:400px; overflow-y:auto;">{highlighted}</div>', unsafe_allow_html=True)
            
            # Professional success message
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b98120, #05966920); border-left: 4px solid #10b981; border-radius: 12px; padding: 1rem; margin-top: 2rem; text-align: center;">
                <span style="font-size: 1.5rem;">✅</span> 
                <span style="font-weight: 600; color: #10b981;">Screening completed successfully</span>
                <br>
                <span style="font-size: 0.9rem; color: #94a3b8;">All resumes ranked by weighted similarity + skill match.</span>
            </div>
            """, unsafe_allow_html=True)