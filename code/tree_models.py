from markov_models import MEMMPredictor

from operator import itemgetter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TreePredictor:
  def __init__(self, num_tags):
    self.word_predictor = MEMMPredictor(num_tags, 0, 1, deleted_features=[], tree_features=True)
    self.vectorizer = DictVectorizer(sparse=False)
    self.clf = LogisticRegression()
    #self.clf = RandomForestClassifier(n_estimators=50)
    
  def tree_features(self, n):
    features = {}
    features['PHRASE' + n.phrase] = 1
    if n.parent:
      features['PARENT' + n.parent.phrase] = 1
      #if n.parent.parent:
      #  features['GRANDPARENT' + n.parent.parent.phrase] = 1
    
      features['ANCESTOR-NP'] = 'NP' in n.parent_phrases
      features['ANCESTOR-VP'] = 'VP' in n.parent_phrases
      features['ANCESTOR-PP'] = 'PP' in n.parent_phrases
      features['ANCESTOR-ADJP'] = 'ADJP' in n.parent_phrases
      features['ANCESTOR-ADVP'] = 'ADVP' in n.parent_phrases
    
    for s in n.siblings:
      features['SIBLING' + s.phrase] = features.get('SIBLING' + s.phrase, 0) + 1

    #features['DEPTH'] = n.depth 
    #features['HEIGHT'] = n.height
    features['SIZE'] = len(n.span)

    return features

  def all_subtrees(self, tree, subtrees):
    subtrees.append(tree)
    for child in tree.children:
      self.all_subtrees(child, subtrees)
    return subtrees

  def get_label(self, tree, labels):
    span_labels = [labels[i] for i in tree.span]
    if all(l == 0 or l == 2 for l in span_labels):
      return 0
    if all(l == 1 or l == 2 for l in span_labels):
      return 1
    return 2

  def subtree_score(self, subtree, word_probas):
    X = self.vectorizer.transform(self.tree_features(subtree))
    tree_probas = self.clf.predict_log_proba(X)[0]
    span_probas = [word_probas[i] for i in subtree.span]
    for word_probas in span_probas:
      tree_probas[0] += (word_probas[0] + word_probas[2])
      tree_probas[1] += (word_probas[1] + word_probas[2])
    return max((tree_probas[0], 0), (tree_probas[1], 1))


  def predict(self, example):
    word_probas = []
    for i in range(example.length):
      word_probas.append(self.word_predictor.get_log_scores(example, i, None))
    labels = [max([l for l in range(3)], key=lambda l: word_probas[i][l]) for i in range(example.length)]
    
    subtrees = self.all_subtrees(example.tree, [])
    subtree_scores = [self.subtree_score(s, word_probas) for s in subtrees]
    span_scores = {}
    for s, score in zip(subtrees, subtree_scores):
      span = (s.span[0], s.span[-1])
      if span in span_scores:
        span_scores[span] = max(span_scores[span], score)
      else:
        span_scores[span]= score

    table = [-1e10] * example.length
    table.append(0) # so table[-1] = 0


    backpointers = [0] * (example.length + 1)
    types = [0] * (example.length + 1)

    for i in range(example.length):
      for j in range(-1, i):
        if (j + 1, i) in span_scores:
          new_score = table[j] + span_scores[(j + 1, i)][0]
          if new_score > table[i]:
            table[i] = new_score
            backpointers[i] = j
            types[i] = span_scores[(j + 1, i)][1]

    best_spans = []
    i = example.length - 1
    while i >= 0:
      j = backpointers[i]
      t = types[i]
      best_spans.append(((j + 1, i), t))
      i = j
    best_spans = [elem for elem in reversed(best_spans)]


    label_restrictions = [0] * example.length
    for span, l in best_spans:
      for i in range(span[0], span[-1] + 1):
        label_restrictions[i] = l

    for i in range(example.length):
      possible_labels = [0, 2]
      if label_restrictions[i] == 1:
        possible_labels = [1, 2]

      labels[i] = max(possible_labels, key=lambda l: word_probas[i][l])
    
    print example.phrase
    print example.labels
    print labels
    print best_spans
    print

    return labels

  def train(self, examples):
    self.word_predictor.train(examples)
    
    y, feature_dicts = [], []
    for example in examples:
      subtrees = self.all_subtrees(example.tree, [])
      for s in subtrees:
        y.append(self.get_label(s, example.labels))
        feature_dicts.append(self.tree_features(s))
        
    X_train = self.vectorizer.fit_transform(feature_dicts)
    self.clf.fit(X_train, y)
