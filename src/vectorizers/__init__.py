from vectorizers.tf_vectorizer import TFVectorizer
from vectorizers.tfidf_vectorizer import TFIDFVectorizer
from vectorizers.lda_vectorizer import LDAVectorizer
from vectorizers.word2vec_vectorizer import Word2VecVectorizer

vectorizers = {
  'tf': TFVectorizer,
  'tfidf': TFIDFVectorizer,
  'lda': LDAVectorizer,
  'w2v': Word2VecVectorizer 
}
