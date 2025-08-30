import streamlit as st
import os
from utils import extract_text_pdf, extract_text_docx, clean_text
from features import build_skill_matcher, extract_skills, extract_years_of_exp
from ranker import combined_score

REQUIRED_SKILLS = ["python", "sql", "aws", "machine learning", "deep learning", "tensorflow"]

st.title("Resume Ranker")
st.write("Upload resumes (PDF/DOCX/TXT) and paste the Job Description to rank candidates.")

job_desc = st.text_area("Job Description", height=150)

uploaded_files = st.file_uploader(
    "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if st.button("Rank Resumes"):

    if not job_desc:
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resumes_texts = []
        resumes_meta = []

        matcher = build_skill_matcher(REQUIRED_SKILLS)

        for file in uploaded_files:
            file_ext = file.name.split(".")[-1].lower()
            raw_text = ""
            if file_ext == "pdf":
                raw_text = extract_text_pdf(file)
            elif file_ext == "docx":
                raw_text = extract_text_docx(file)
            elif file_ext == "txt":
                raw_text = str(file.read(), "utf-8")
            text = clean_text(raw_text)
            resumes_texts.append(text)

            resumes_meta.append({
                "skills": extract_skills(text, matcher),
                "years": extract_years_of_exp(text)
            })
        scores = combined_score(job_desc, resumes_texts, resumes_meta, REQUIRED_SKILLS)
        results = sorted(zip([f.name for f in uploaded_files], scores), key=lambda x: x[1], reverse=True)
        st.subheader("Ranked Resumes:")
        for i, (name, score) in enumerate(results, 1):
            st.write(f"{i}. {name} — Fit Score: {round(score*100, 2)}%")
