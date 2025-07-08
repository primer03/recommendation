# vector.py
from pythainlp import word_vector
from pythainlp.tokenize import word_tokenize
import numpy as np

model = word_vector.WordVector("thai2fit_wv").get_model()

def get_vector(text: str):
    tokens = word_tokenize(text, keep_whitespace=False)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0).tolist() if vectors else [0]*model.vector_size
