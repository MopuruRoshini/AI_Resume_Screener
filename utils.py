import pdfplumber
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from collections import Counter

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ---------- Load spaCy model (auto-download) ----------
nlp = spacy.load("en_core_web_sm")

# ---------- PDF text extraction ----------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------- Preprocessing for similarity ----------
def preprocess_resume(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ---------- Dynamic keyword extraction ----------
def extract_keywords(text, top_n=15):
    doc = nlp(text)
    keywords = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if len(phrase.split()) <= 3 and len(phrase) > 2:
            keywords.append(phrase)
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 2:
            keywords.append(token.lemma_.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]
    freq = Counter(filtered)
    return set([kw for kw, _ in freq.most_common(top_n)])

# ---------- Main screening function ----------
def compute_similarity_and_skills(job_desc, resumes, custom_skills=None):
    job_processed = preprocess_resume(job_desc)
    jd_keywords = extract_keywords(job_desc, top_n=20)
    if custom_skills:
        jd_keywords = jd_keywords.union(set(custom_skills))
    
    corpus = [job_processed] + [preprocess_resume(text) for _, text in resumes]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    
    results = []
    for (filename, raw_text), score in zip(resumes, similarities):
        resume_keywords = extract_keywords(raw_text, top_n=15)
        top_skills = ", ".join(list(resume_keywords)[:10]) if resume_keywords else "None"
        matched = jd_keywords.intersection(resume_keywords)
        missing = jd_keywords.difference(resume_keywords)
        skill_match_percent = (len(matched) / len(jd_keywords) * 100) if jd_keywords else 0
        results.append({
            "Resume": filename,
            "Similarity Score": round(score, 4),
            "Skill Match %": round(skill_match_percent, 1),
            "Top Resume Skills": top_skills,
            "Matched Skills": ", ".join(sorted(matched)) if matched else "None",
            "Missing Skills": ", ".join(sorted(missing)) if missing else "None"
        })
    return pd.DataFrame(results)

# ---------- Resume-to-resume comparison ----------
def compare_two_resumes(text1, text2):
    proc1 = preprocess_resume(text1)
    proc2 = preprocess_resume(text2)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([proc1, proc2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]