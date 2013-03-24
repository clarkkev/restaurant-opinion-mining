from rule_based import BaselinePredictor, TreeBaselinePredictor
from markov_models import HMMPredictor, MEMMPredictor
from loader import Loader

def get_metrics(predictions, examples, target):
  tp = fp = fn = 0
  for prediction, example in zip(predictions, examples):
    for predicted, actual in zip(prediction, example.labels):
      if predicted == target:
        if predicted == actual:
          tp += 1
        else:
          fp += 1
      elif predicted != actual:
          fn += 1
  
    print example.tagged
    print example.labels
    print prediction
    print 
  
  print tp, fn
  p = float(tp) / (tp + fp)
  r = float(tp) / (tp + fn)
  f1 = (2 * p * r) / (p + r)

  return p, r, f1

def evaluate(predictor, train, test):
  predictor.train(train)
  predictions = [predictor.predict(example) for example in test]
  print get_metrics(predictions, test, 0)
  print get_metrics(predictions, test, 1)

if __name__ == '__main__':
  loader = Loader()
  #for tag in sorted(list(loader.tagset)):
  #  print tag + ": ,"
  #evaluate(HMMPredictor(loader.num_tags), loader.train, loader.validation)
  evaluate(MEMMPredictor(loader.num_tags), loader.train, loader.validation)
  #evaluate(MEMMPredictor(loader.num_tags), loader.train, loader.test)
  #evaluate(BaselinePredictor(['NN'], ['JJ']), loader.train, loader.examples)
