import spacy
from spacy.matcher import PhraseMatcher
import re

nlp = spacy.load("en_core_web_sm")

def build_skill_matcher(skills):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s) for s in skills]
    matcher.add("SKILLS", patterns)
    return matcher

def extract_skills(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for _, start, end in matches:
        found.add(doc[start:end].text.lower())
    return list(found)

def extract_years_of_exp(text):
    m = re.search(r'(\d+)\+?\s*(years|yrs)', text.lower())
    if m:
        return int(m.group(1))
    return 0
