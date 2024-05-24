from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer():
  def __init__(self, params):
    self.vectorizer = TfidfVectorizer(**params.get(params.get('vectorizer')))
  
  def fit_transform(self, docs):
    return self.vectorizer.fit_transform(docs)
