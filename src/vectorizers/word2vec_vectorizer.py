from gensim.models import Word2Vec
import numpy as np


class Word2VecVectorizer():
  def __init__(self, params):
    self.params = params

  def transform(self, doc, vectorizer, n_dimensions):
    num_tokens = 0
    doc_vec = np.zeros((n_dimensions,), dtype='float32')


    key_to_index = vectorizer.wv.key_to_index
    for token in doc:
      if token in key_to_index.keys():
        num_tokens += 1
        doc_vec = np.add(doc_vec, vectorizer.wv.vectors[key_to_index.get(token)])
    
    return np.divide(doc_vec, num_tokens)


  def fit_transform(self, docs):
    vectorizer = Word2Vec(sentences=docs, **self.params.get(self.params.get('vectorizer')))
    n_dimensions = len(vectorizer.wv.vectors[0])

    return [
      self.transform(doc, vectorizer, n_dimensions) for doc in docs
    ]
