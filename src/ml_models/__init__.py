from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

models = {
  'svc': LinearSVC,
  'random_forest': RandomForestClassifier,
  'naive_bayes': MultinomialNB
}

# TODO: mark time
# TODO: save results
# TODO: create plot
def run_model(model, X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train)
  predicted = model.predict(X_test)

  cv_scores = cross_val_score(model, X_train, y_train, cv=5)

  return classification_report(y_test, predicted, output_dict=True), cv_scores
