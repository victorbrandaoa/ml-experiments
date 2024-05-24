from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_base_vectorizer(base_vectorizer, params):
  return CountVectorizer(**params) if base_vectorizer == 'tf' else TfidfVectorizer(**params)


class LDAVectorizer():
  def __init__(self, params):
    self.vectorizer = build_base_vectorizer(params.get('base_vectorizer'), params.get(params.get('base_vectorizer')))
    self.lda = LatentDirichletAllocation(**params.get('lda'))

  def fit_transform(self, docs):
    vectorized_docs = self.vectorizer.fit_transform(docs)
    return self.lda.fit_transform(vectorized_docs)
