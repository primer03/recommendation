# utils.py
from bs4 import BeautifulSoup

from sklearn.metrics.pairwise import cosine_similarity

def get_clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

