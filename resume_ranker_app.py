# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import hstack
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import PyPDF2
import docx
import tempfile
import os

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    st.warning(f"NLTK download warning: {e}")

# Set page configuration
st.set_page_config(
    page_title="Resume Ranker Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        color: #1e293b;
        margin: 2rem 0 1.5rem 0;
        font-weight: 700;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 5px;
        height: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
    }
    
    /* Score Cards */
    .score-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .score-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .score-card h3 {
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .score-card h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Badge Styles */
    .keyword-match, .skill-match {
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        margin: 0.4rem;
        display: inline-block;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .keyword-match {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .keyword-match:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
    }
    
    .skill-match {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .skill-match:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* File Upload Box */
    .file-upload-box {
        border: 3px dashed #cbd5e1;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        background: white;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .file-upload-box:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-color: #667eea;
        border-style: solid;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .metric-card h4 {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h3, .metric-card h2 {
        font-size: 1.5rem;
        color: #1e293b;
        margin: 0;
        font-weight: 700;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    .strong-match {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .good-match {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .weak-match {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border-radius: 8px;
        padding: 0 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748b;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] .section-header {
        font-size: 1.25rem;
        margin-top: 1rem;
    }
    
    /* DataFrame Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px;
        padding: 1.25rem;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: none;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid #e2e8f0;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        font-weight: 600;
        color: #1e293b;
        font-size: 1rem;
    }
    
    /* Selectbox */
    .stSelectbox > label {
        font-weight: 600;
        color: #1e293b;
        font-size: 1rem;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Subheader Styling */
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    }
</style>
""", unsafe_allow_html=True)

# [Keep all the preprocessing functions exactly as they are]
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text is None:
        return ""

    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        except:
            tokens = [token for token in tokens if len(token) > 2]

        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            pass

        return ' '.join(tokens)
    
    except Exception as e:
        st.error(f"Error in text preprocessing: {e}")
        return ""

def create_additional_features(df):
    """Create additional features from resume text"""
    features = pd.DataFrame(index=df.index)

    try:
        features['text_length'] = df['cleaned_resume'].str.len().fillna(0)
        features['word_count'] = df['cleaned_resume'].str.split().str.len().fillna(0)

        key_skills = ['python', 'java', 'machine', 'learning', 'sql', 'javascript',
                      'react', 'docker', 'aws', 'data', 'analysis', 'development',
                      'tensorflow', 'pytorch', 'deep', 'neural', 'network', 'cloud',
                      'kubernetes', 'linux', 'database', 'api', 'web', 'mobile',
                      'software', 'engineer', 'developer', 'scientist', 'analyst']

        for skill in key_skills:
            features[f'skill_{skill}'] = df['cleaned_resume'].str.contains(skill, na=False).astype(int)

        exp_pattern = r'(\d+)\s+years?'
        features['experience'] = df['Resume_str'].str.extract(exp_pattern, expand=False).fillna(0).astype(int)

    except Exception as e:
        st.error(f"Error creating additional features: {e}")
    
    return features

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        with open(tmp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        os.unlink(tmp_path)
        return text.strip()
    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(docx_file.getvalue())
            tmp_path = tmp_file.name
        
        doc = docx.Document(tmp_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
        
        os.unlink(tmp_path)
        return text.strip()
    
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text"""
    try:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        if file_type == "application/pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        else:
            try:
                return str(uploaded_file.read(), "utf-8")
            except:
                st.error(f"Unsupported file format: {file_type}")
                return ""
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return ""

class ResumeRanker:
    def __init__(self, model, tfidf_vectorizer, feature_engineer, additional_feature_columns):
        self.model = model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.feature_engineer = feature_engineer
        self.additional_feature_columns = additional_feature_columns
        self.key_skills = ['python', 'java', 'machine', 'learning', 'sql', 'javascript',
                          'react', 'docker', 'aws', 'data', 'analysis', 'development',
                          'tensorflow', 'pytorch', 'deep', 'neural', 'network', 'cloud',
                          'kubernetes', 'linux', 'database', 'api', 'web', 'mobile',
                          'software', 'engineer', 'developer', 'scientist', 'analyst']

    def preprocess_resume(self, resume_text):
        return preprocess_text(resume_text)

    def extract_features(self, resume_text):
        try:
            cleaned_text = self.preprocess_resume(resume_text)
            tfidf_feats = self.tfidf_vectorizer.transform([cleaned_text])

            temp_df = pd.DataFrame([{'cleaned_resume': cleaned_text, 'Resume_str': resume_text}])
            additional_feats = self.feature_engineer(temp_df)
            
            for col in self.additional_feature_columns:
                if col not in additional_feats.columns:
                    additional_feats[col] = 0
            
            additional_feats = additional_feats[self.additional_feature_columns]
            additional_feats = additional_feats.fillna(0)
            
            additional_feats_sparse = additional_feats.astype(np.float64).sparse.to_coo().tocsr()
            combined_feats = hstack([tfidf_feats, additional_feats_sparse])
            return combined_feats
        
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return hstack([self.tfidf_vectorizer.transform([""]), 
                          np.zeros((1, len(self.additional_feature_columns)))])

    def predict_fit_score(self, resume_text):
        try:
            features = self.extract_features(resume_text)
            probability = self.model.predict_proba(features)[0, 1]
            return probability
        except Exception as e:
            st.error(f"Error predicting fit score: {e}")
            return 0.0

    def analyze_keywords(self, resume_text):
        """Analyze keyword matches in the resume"""
        try:
            cleaned_text = self.preprocess_resume(resume_text)
            words = set(cleaned_text.split())
            
            matched_keywords = [skill for skill in self.key_skills if skill in words]
            keyword_score = len(matched_keywords) / len(self.key_skills) if self.key_skills else 0
            
            return {
                'matched_keywords': matched_keywords,
                'keyword_score': keyword_score,
                'total_keywords': len(self.key_skills),
                'matched_count': len(matched_keywords)
            }
        except Exception as e:
            st.error(f"Error in keyword analysis: {e}")
            return {
                'matched_keywords': [],
                'keyword_score': 0.0,
                'total_keywords': len(self.key_skills),
                'matched_count': 0
            }

    def analyze_skills(self, resume_text):
        """Analyze technical skills in the resume"""
        try:
            cleaned_text = self.preprocess_resume(resume_text)
            
            tech_skills = {
                'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust'],
                'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask'],
                'Data Science': ['machine', 'learning', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy'],
                'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
                'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git'],
                'Tools': ['excel', 'tableau', 'powerbi', 'jupyter', 'vscode', 'pycharm']
            }
            
            matched_skills = {}
            skill_scores = {}
            
            for category, skills in tech_skills.items():
                matched = [skill for skill in skills if skill in cleaned_text]
                matched_skills[category] = matched
                skill_scores[category] = len(matched) / len(skills) if skills else 0
            
            overall_skill_score = np.mean(list(skill_scores.values())) if skill_scores else 0
            
            return {
                'matched_skills': matched_skills,
                'skill_scores': skill_scores,
                'overall_skill_score': overall_skill_score
            }
        except Exception as e:
            st.error(f"Error in skill analysis: {e}")
            return {
                'matched_skills': {},
                'skill_scores': {},
                'overall_skill_score': 0.0
            }

    def get_detailed_analysis(self, resume_text):
        """Get comprehensive analysis of the resume"""
        try:
            fit_score = self.predict_fit_score(resume_text)
            keyword_analysis = self.analyze_keywords(resume_text)
            skill_analysis = self.analyze_skills(resume_text)
            
            overall_score = (
                fit_score * 0.5 + 
                keyword_analysis['keyword_score'] * 0.3 + 
                skill_analysis['overall_skill_score'] * 0.2
            )
            
            return {
                'fit_score': fit_score,
                'keyword_analysis': keyword_analysis,
                'skill_analysis': skill_analysis,
                'overall_score': overall_score,
                'word_count': len(resume_text.split()),
                'experience_years': self.extract_experience(resume_text)
            }
        except Exception as e:
            st.error(f"Error in detailed analysis: {e}")
            return {
                'fit_score': 0.0,
                'keyword_analysis': {'keyword_score': 0.0, 'matched_keywords': [], 'matched_count': 0},
                'skill_analysis': {'overall_skill_score': 0.0, 'matched_skills': {}, 'skill_scores': {}},
                'overall_score': 0.0,
                'word_count': 0,
                'experience_years': 0
            }
    
    def extract_experience(self, resume_text):
        """Extract years of experience from resume text"""
        try:
            exp_pattern = r'(\d+)\s+years?'
            matches = re.findall(exp_pattern, resume_text.lower())
            if matches:
                return max([int(match) for match in matches])
            return 0
        except:
            return 0

    def rank_resumes(self, resumes):
        """Rank multiple resumes"""
        rankings = []
        for i, resume in enumerate(resumes):
            analysis = self.get_detailed_analysis(resume)
            rankings.append({
                'resume_id': i,
                'resume_preview': resume[:100] + '...',
                'analysis': analysis
            })
        
        rankings.sort(key=lambda x: x['analysis']['overall_score'], reverse=True)
        return rankings

@st.cache_resource
def load_model():
    try:
        loaded_components = joblib.load('resume_ranker_model.pkl')
        ranker = ResumeRanker(
            loaded_components['model'],
            loaded_components['tfidf_vectorizer'],
            loaded_components['feature_engineer'],
            loaded_components['additional_feature_columns']
        )
        return ranker
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def main():
    st.markdown('<h1 class="main-header">Resume Ranker Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Resume Screening and Ranking System</p>', unsafe_allow_html=True)
    
    ranker = load_model()
    if ranker is None:
        st.error("Failed to load the model. Please ensure 'resume_ranker_model.pkl' is in the current directory.")
        return

    with st.sidebar:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        
        input_method = st.radio("Choose input method:", 
                                       ["Single Resume", "File Upload", "Batch Upload", "Sample Test"])
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.info("""
        **Single Resume**: Paste text or upload one file
        **File Upload**: Upload multiple PDF/DOCX files
        **Batch Upload**: Upload CSV with resume texts
        **Sample Test**: Test with sample resumes
        """)

    if input_method == "Single Resume":
        st.markdown('<div class="section-header">Single Resume Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Paste Text")
            resume_text = st.text_area("Paste resume text here:", height=200,
                                     placeholder="Enter the resume content here...",
                                     label_visibility="collapsed")
            
            if st.button("Analyze Pasted Resume", type="primary", key="paste_analyze", use_container_width=True):
                if resume_text.strip():
                    with st.spinner("Analyzing resume..."):
                        analysis = ranker.get_detailed_analysis(resume_text)
                        st.session_state.analysis_results = [analysis]
                        st.session_state.resumes = [resume_text]
                else:
                    st.warning("Please enter resume text.")
        
        with col2:
            st.subheader("Upload File")
            uploaded_file = st.file_uploader("Upload resume file", 
                                           type=['pdf', 'docx', 'txt'],
                                           key="single_upload",
                                           label_visibility="collapsed")
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                if st.button("Analyze Uploaded File", type="primary", key="file_analyze", use_container_width=True):
                    with st.spinner("Extracting text and analyzing..."):
                        resume_text = process_uploaded_file(uploaded_file)
                        
                        if resume_text.strip():
                            analysis = ranker.get_detailed_analysis(resume_text)
                            st.session_state.analysis_results = [analysis]
                            st.session_state.resumes = [resume_text]
                            st.session_state.uploaded_files = [uploaded_file.name]
                        else:
                            st.error("Could not extract text from the file.")
    
    elif input_method == "File Upload":
        st.markdown('<div class="section-header">Multiple File Upload</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
        st.markdown('### Drag and Drop Files')
        st.markdown('Supported formats: PDF, DOCX, TXT')
        uploaded_files = st.file_uploader("", 
                                        type=['pdf', 'docx', 'txt'],
                                        accept_multiple_files=True,
                                        key="multi_upload",
                                        label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            with st.expander("View Uploaded Files", expanded=True):
                for file in uploaded_files:
                    st.write(f"• {file.name}")
            
            if st.button("Analyze All Uploaded Files", type="primary", use_container_width=True):
                with st.spinner("Processing files..."):
                    resumes = []
                    file_names = []
                    successful_files = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        resume_text = process_uploaded_file(uploaded_file)
                        if resume_text.strip():
                            resumes.append(resume_text)
                            file_names.append(uploaded_file.name)
                            successful_files += 1
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if successful_files > 0:
                        analyses = []
                        status_text.text("Analyzing resumes...")
                        progress_bar2 = st.progress(0)
                        
                        for i, resume in enumerate(resumes):
                            analyses.append(ranker.get_detailed_analysis(resume))
                            progress_bar2.progress((i + 1) / len(resumes))
                        
                        st.session_state.analysis_results = analyses
                        st.session_state.resumes = resumes
                        st.session_state.uploaded_files = file_names
                        
                        status_text.text("")
                        st.success(f"Successfully processed {successful_files} out of {len(uploaded_files)} files")
                    else:
                        st.error("Could not extract text from any of the uploaded files.")
    
    elif input_method == "Batch Upload":
        st.markdown('<div class="section-header">Batch CSV Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file with resumes", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"CSV file loaded with {len(df)} rows")
                
                st.write("**Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                text_column = st.selectbox("Select the column containing resume text:", 
                                         df.columns.tolist())
                
                if st.button("Analyze All Resumes", type="primary", use_container_width=True):
                    with st.spinner("Analyzing resumes..."):
                        resumes = df[text_column].dropna().tolist()
                        st.session_state.resumes = resumes
                        analyses = []
                        
                        progress_bar = st.progress(0)
                        for i, resume in enumerate(resumes):
                            analyses.append(ranker.get_detailed_analysis(resume))
                            progress_bar.progress((i + 1) / len(resumes))
                            
                        st.session_state.analysis_results = analyses
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:
        st.markdown('<div class="section-header">Sample Resume Test</div>', unsafe_allow_html=True)
        sample_resumes = [
            "Experienced Data Scientist with 5 years in machine learning and Python. Strong background in deep learning, TensorFlow, and statistical analysis. PhD in Computer Science. Skills include SQL, AWS, and data visualization.",
            "Junior web developer with 1 year experience in HTML, CSS, and basic JavaScript. Bachelor's degree in IT. Familiar with React and Node.js.",
            "Software Engineer proficient in Java and Python. 3 years experience in backend development and API design. Strong algorithms knowledge. Master's degree in Computer Science. Experience with Docker and cloud platforms.",
            "Data Analyst with SQL, Excel, and Tableau skills. 2 years experience in business intelligence and data visualization. Bachelor's degree in Business Analytics. Knowledge of Python for data analysis."
        ]
        
        selected_sample = st.selectbox("Choose a sample resume:", 
                                      ["Sample 1: Senior Data Scientist", 
                                       "Sample 2: Junior Web Developer",
                                       "Sample 3: Software Engineer",
                                       "Sample 4: Data Analyst"])
        
        sample_index = ["Sample 1", "Sample 2", "Sample 3", "Sample 4"].index(selected_sample.split(":")[0])
        
        if st.button("Test with Sample", type="primary", use_container_width=True):
            with st.spinner("Analyzing sample resume..."):
                analysis = ranker.get_detailed_analysis(sample_resumes[sample_index])
                st.session_state.analysis_results = [analysis]
                st.session_state.resumes = [sample_resumes[sample_index]]

    if st.session_state.analysis_results:
        display_results(ranker, st.session_state.analysis_results, st.session_state.resumes)

def display_results(ranker, analyses, resumes):
    st.markdown("---")
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    
    if len(analyses) == 1:
        display_single_analysis(ranker, analyses[0], resumes[0])
    else:
        display_ranking(ranker, analyses, resumes)

def display_single_analysis(ranker, analysis, resume_text):
    st.markdown("### Overall Scores")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h3>Overall Score</h3>
            <h2>{analysis['overall_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h3>Fit Score</h3>
            <h2>{analysis['fit_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="score-card">
            <h3>Keyword Match</h3>
            <h2>{analysis['keyword_analysis']['keyword_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="score-card">
            <h3>Skills Score</h3>
            <h2>{analysis['skill_analysis']['overall_skill_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Detailed Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Breakdown", "Keywords", "Skills", "Resume Details", "Extracted Text"])
    
    with tab1:
        display_score_breakdown(analysis)
    
    with tab2:
        display_keyword_analysis(analysis['keyword_analysis'])
    
    with tab3:
        display_skill_analysis(analysis['skill_analysis'])
    
    with tab4:
        display_resume_details(resume_text, analysis)
    
    with tab5:
        display_extracted_text(resume_text)

def display_score_breakdown(analysis):
    col1, col2 = st.columns(2)
    
    with col1:
        scores = {
            'Fit Score': analysis['fit_score'],
            'Keyword Match': analysis['keyword_analysis']['keyword_score'],
            'Skills Score': analysis['skill_analysis']['overall_skill_score']
        }
        
        fig = px.pie(
            values=list(scores.values()),
            names=list(scores.keys()),
            title="Score Distribution",
            color=list(scores.keys()),
            color_discrete_map={
                'Fit Score': '#667eea',
                'Keyword Match': '#10b981',
                'Skills Score': '#3b82f6'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        skill_categories = list(analysis['skill_analysis']['skill_scores'].keys())
        skill_scores = list(analysis['skill_analysis']['skill_scores'].values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=skill_scores + [skill_scores[0]],
            theta=skill_categories + [skill_categories[0]],
            fill='toself',
            line=dict(color='#667eea', width=2),
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Skill Category Scores",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Additional Metrics")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Years of Experience</h4>
            <h2>{analysis['experience_years']} years</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Word Count</h4>
            <h2>{analysis['word_count']} words</h2>
        </div>
        """, unsafe_allow_html=True)

def display_keyword_analysis(keyword_analysis):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Keyword Statistics")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Matched Keywords</h4>
            <h2>{keyword_analysis['matched_count']}/{keyword_analysis['total_keywords']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Keyword Match Score</h4>
            <h2>{keyword_analysis['keyword_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Matched Keywords")
        if keyword_analysis['matched_keywords']:
            for keyword in keyword_analysis['matched_keywords']:
                st.markdown(f'<span class="keyword-match">{keyword.upper()}</span>', unsafe_allow_html=True)
        else:
            st.info("No keywords matched.")
    
    with col2:
        fig = go.Figure(go.Bar(
            x=['Matched', 'Missing'],
            y=[keyword_analysis['matched_count'], 
               keyword_analysis['total_keywords'] - keyword_analysis['matched_count']],
            marker_color=['#10b981', '#ef4444']
        ))
        fig.update_layout(
            title="Keyword Match Overview",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def display_skill_analysis(skill_analysis):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Skills by Category")
        for category, skills in skill_analysis['matched_skills'].items():
            score = skill_analysis['skill_scores'][category]
            
            with st.expander(f"**{category}** (Score: {score:.1%})", expanded=True):
                if skills:
                    for skill in skills:
                        st.markdown(f'<span class="skill-match">{skill.upper()}</span>', unsafe_allow_html=True)
                else:
                    st.info("No skills matched in this category")
    
    with col2:
        st.markdown("#### Skills Overview")
        if skill_analysis['skill_scores']:
            categories = list(skill_analysis['skill_scores'].keys())
            scores = list(skill_analysis['skill_scores'].values())
            
            fig = px.bar(
                x=scores,
                y=categories,
                title="Skills Score by Category",
                labels={'x': 'Score', 'y': 'Category'},
                color=scores,
                color_continuous_scale='Viridis',
                orientation='h'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_resume_details(resume_text, analysis):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Analysis Summary")
        
        summary_data = {
            'Metric': ['Overall Score', 'Fit Probability', 'Keyword Match', 
                      'Skills Score', 'Experience', 'Word Count'],
            'Value': [f"{analysis['overall_score']:.1%}", 
                     f"{analysis['fit_score']:.1%}", 
                     f"{analysis['keyword_analysis']['keyword_score']:.1%}",
                     f"{analysis['skill_analysis']['overall_skill_score']:.1%}",
                     f"{analysis['experience_years']} years",
                     f"{analysis['word_count']} words"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Recommendation")
        if analysis['overall_score'] >= 0.7:
            st.markdown('<div class="recommendation-card strong-match"><strong>Strong Match</strong> - This candidate is highly recommended!</div>', unsafe_allow_html=True)
        elif analysis['overall_score'] >= 0.5:
            st.markdown('<div class="recommendation-card good-match"><strong>Good Match</strong> - This candidate is worth considering.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-card weak-match"><strong>Weak Match</strong> - This candidate may not be the best fit.</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Quick Stats")
        
        stats_data = [
            ("Keywords Found", f"{analysis['keyword_analysis']['matched_count']}"),
            ("Skill Categories", f"{len([s for s in analysis['skill_analysis']['skill_scores'].values() if s > 0])}"),
            ("Total Skills", f"{sum(len(skills) for skills in analysis['skill_analysis']['matched_skills'].values())}"),
            ("Content Quality", "High" if analysis['word_count'] > 200 else "Medium")
        ]
        
        for title, value in stats_data:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <h3>{value}</h3>
            </div>
            """, unsafe_allow_html=True)

def display_extracted_text(resume_text):
    st.markdown("#### Extracted Text")
    st.text_area("", resume_text, height=400, key="extracted_text", label_visibility="collapsed")

def display_ranking(ranker, analyses, resumes):
    st.markdown("#### Resume Rankings")
    
    ranking_data = []
    for i, (analysis, resume) in enumerate(zip(analyses, resumes)):
        file_name = st.session_state.uploaded_files[i] if i < len(st.session_state.uploaded_files) else f"Resume {i+1}"
        
        ranking_data.append({
            'Rank': i + 1,
            'File Name': file_name,
            'Overall Score': f"{analysis['overall_score']:.1%}",
            'Fit Score': f"{analysis['fit_score']:.1%}",
            'Keyword Score': f"{analysis['keyword_analysis']['keyword_score']:.1%}",
            'Skills Score': f"{analysis['skill_analysis']['overall_skill_score']:.1%}",
            'Experience': f"{analysis['experience_years']} years",
            'Word Count': analysis['word_count'],
            'Preview': resume[:100] + '...'
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    
    csv = ranking_df.to_csv(index=False)
    st.download_button(
        label="Download Ranking Results",
        data=csv,
        file_name="resume_rankings.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("#### Score Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        scores = [analysis['overall_score'] for analysis in analyses]
        fig = px.histogram(x=scores, nbins=20, title="Overall Score Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_scores = sorted([analysis['overall_score'] for analysis in analyses], reverse=True)[:10]
        fig = px.bar(x=range(1, len(top_scores) + 1), y=top_scores, 
                    title="Top 10 Resume Scores")
        fig.update_layout(
            xaxis_title="Rank", 
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()