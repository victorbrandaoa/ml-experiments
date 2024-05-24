from sklearn.feature_extraction.text import CountVectorizer


class TFVectorizer():
  def __init__(self, params):
    self.vectorizer = CountVectorizer(**params.get(params.get('vectorizer')))
  
  def fit(self, docs):
    self.vectorizer.fit(docs)

  def transform(self, docs):
    return self.vectorizer.transform(docs)
