from utilis import clean_text
from features import build_skill_matcher, extract_skills, extract_years_of_exp
from ranker import combined_score
skills = ["python", "sql", "aws", "machine learning", "deep learning", "tensorflow"]

# Sample Job Description
job_desc = "Looking for a Data Scientist with skills in Python, SQL, Machine Learning, AWS"

# Sample resumes text
resumes_texts = [
    "Alice: Experienced in Python, SQL, AWS, machine learning",
    "Bob: Skilled in Java, C++, Linux",
    "Charlie: Python, deep learning, cloud (AWS, Azure)"
]

required_skills = ["python", "sql", "aws", "machine learning"]

# Build matcher & extract features
matcher = build_skill_matcher(required_skills)
resumes_meta = []
for text in resumes_texts:
    resumes_meta.append({
        'skills': extract_skills(text, matcher),
        'years': extract_years_of_exp(text)
    })

# Compute scores
scores = combined_score(job_desc, resumes_texts, resumes_meta, required_skills)
for r, s in zip(resumes_texts, scores):
    print(f"{r[:30]}... -> Score: {round(s*100,2)}")
