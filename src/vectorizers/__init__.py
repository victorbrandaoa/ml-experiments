from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vectorizers.lda_vectorizer import LDAVectorizer
from vectorizers.word2vec_vectorizer import Word2VecVectorizer

vectorizers = {
  'tf': CountVectorizer,
  'tfidf': TfidfVectorizer,
  'lda': LDAVectorizer,
  'w2v': Word2VecVectorizer 
}
