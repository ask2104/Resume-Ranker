from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def rank_by_tfidf(job_description, resumes_texts):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
    docs = [job_description] + resumes_texts
    tfidf = vectorizer.fit_transform(docs)
    job_vec = tfidf[0]
    resume_vecs = tfidf[1:]
    sims = cosine_similarity(job_vec, resume_vecs)[0]
    return sims

def combined_score(job_description, resumes_texts, resumes_meta, required_skills, weights=(0.6,0.3,0.1)):
    sims = rank_by_tfidf(job_description, resumes_texts)
    skill_scores = []
    year_scores = []
    for meta in resumes_meta:
        skill_scores.append(len(set(meta['skills']) & set(required_skills))/max(1,len(required_skills)))
        year_scores.append(meta.get('years',0))
    comp = np.vstack([sims, skill_scores, year_scores]).T
    comp_scaled = MinMaxScaler().fit_transform(comp)
    final = comp_scaled.dot(np.array(weights))
    return final
