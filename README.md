Resume Ranker

A Python-based project that ranks candidates’ resumes based on their fit to a given Job Description (JD) using NLP techniques.

Features

Parse resumes (PDF, DOCX, TXT)

Extract key info: skills, experience

Compute similarity between resumes and JD using TF-IDF + weighted scoring

Rank candidates with a fit score

Highlight matched and missing skills for explainability

Optional Streamlit web demo for interactive ranking

Setup
git clone https://github.com/<your-username>/resume-ranker.git
cd resume-ranker
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Usage

Test Script: python src/test_ranker.py

Streamlit Demo: streamlit run src/app.py
