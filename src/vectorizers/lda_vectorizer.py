from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_base_vectorizer(base_vectorizer, params):
  return CountVectorizer(**params) if base_vectorizer == 'tf' else TfidfVectorizer(**params)


class LDAVectorizer():
  def __init__(self, **kwargs):
    self.vectorizer = build_base_vectorizer(kwargs.get('base_vectorizer'), kwargs.get(kwargs.get('base_vectorizer')))
    self.lda = LatentDirichletAllocation(**kwargs.get('params'))

  def fit(self, docs):
    vectorized_docs = self.vectorizer.fit_transform(docs)
    self.lda.fit(vectorized_docs)
  
  def transform(self, docs):
    vectorized_docs = self.vectorizer.transform(docs)
    return self.lda.transform(vectorized_docs)
