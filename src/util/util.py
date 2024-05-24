import yaml
from util.constants import CONFIG_FILE
from sklearn.model_selection import ParameterGrid


def read_config():
  with open(CONFIG_FILE) as file:
    data = yaml.safe_load(file.read())
  
  return data


def format_config(model_name, model_type, combs):
  return list(map(lambda comb: { model_type: model_name, model_name: comb }, combs))


def format_lda_vectorizer_config(base_vectorizer, base_vectorizer_combs, lda_combs):
  combs = list(ParameterGrid(
    {
      'base_vectorizer': [base_vectorizer],
      base_vectorizer: base_vectorizer_combs,
      'lda': lda_combs
    }
  ))

  return list(map(
    lambda comb: { 
      'vectorizer': 'lda',
      **comb
    }, combs)
  )


def gen_vectorizers_combinations(params):
  combs = []
  for v in params.get('vectorizers'):
    if v == 'lda':
      for base_v in params.get(v).get('base_vectorizer'):
        combs.extend(format_lda_vectorizer_config(base_v, list(ParameterGrid(params.get(base_v))), list(ParameterGrid(params.get(v).get('params')))))
    else:
      combs.extend(format_config(v, 'vectorizer', list(ParameterGrid(params.get(v)))))
  
  return combs


def gen_classifiers_combinations(params):
  combs = []
  for clf in params.get('classifiers'):
    combs.extend(format_config(clf, 'classifier', list(ParameterGrid(params.get(clf)))))
  
  return combs
  