from util import read_config, gen_vectorizers_combinations, ROOT_DIR, gen_classifiers_combinations
from preprocess import preprocess_func
from vectorizers import vectorizers
from sklearn.model_selection import train_test_split
import pandas as pd
from ml_models import models, run_model

def main():
  data = read_config()
  df = pd.read_csv(f'{ROOT_DIR}/../ml_dataset.csv')
  X_train, X_test, y_train, y_test = train_test_split(df['content'], df['pii_count'], test_size=0.2, random_state=42)

  for comb in gen_vectorizers_combinations(data):
    vectorizer = vectorizers.get(comb.get('vectorizer'))(**comb.get(comb.get('vectorizer')))
    vectorizer.fit(X_train)
    X_train_emb = vectorizer.transform(X_train)
    X_test_emb = vectorizer.transform(X_test)
    for clf_comb in gen_classifiers_combinations(data):
      clf_name = clf_comb.get('classifier')
      model = models.get(clf_comb.get('classifier'))(**clf_comb.get(clf_name))
      report, cv_score = run_model(model, X_train_emb, y_train, X_test_emb, y_test)
      print(f'{comb} && {clf_comb}')
      print(f'Report:\n{report}')
      print(f'CV Score: {cv_score.mean()}')


if __name__ == '__main__':
  main()
