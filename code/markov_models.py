from collections import Counter
from math import log
from itertools import product
from loader import Example

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

class MarkovModel:
  def __init__(self, num_tags, label_depth, tag_depth):
    self.num_tags = num_tags
    self.label_depth = label_depth
    self.tag_depth = tag_depth

  def predict(self, example):
    table = [[-1.0e15 for i in range(3)] for j in range(example.length)]

    for i in range(example.length):
      for previous_labels in product(range(3), repeat = min(self.label_depth, i)):
        score = 0
        for j in range(len(previous_labels)):
          score += table[i - j - 1][previous_labels[len(previous_labels) - j - 1]]

        ps = self.get_log_scores(example, i, previous_labels)
        for label in range(3):
          table[i][label] = max(table[i][label], score + ps[label])

    labels = []
    for i in range(example.length):
      labels.append(max(range(3), key=lambda label: table[i][label]))
    return labels

  def print_table(self, table):
    for label in range(3):
      row = ""
      for i in range(len(table)):
        row += "{:<8}".format(int(100 * table[i][label]))
      print row

class MEMMPredictor(MarkovModel):
  def __init__(self, num_tags, label_depth, tag_depth, deleted_features=[], tree_features=False):
    MarkovModel.__init__(self, num_tags, label_depth, tag_depth)
    self.vectorizer = DictVectorizer(sparse=False)
    self.clf = LogisticRegression()
    self.tree_features = tree_features
    self.deleted_features = deleted_features
  
  def get_log_scores(self, example, i, previous_labels):
    features = self.get_features(example, i, previous_labels)
    return self.clf.predict_log_proba(self.vectorizer.transform([features]))[0]

  def get_features(self, example, i, previous_labels):
    previous_tags = example.tags[max(i - self.tag_depth, 0):i]
    
    features = {}
    for j in range(self.label_depth):
      if j >= len(previous_labels):
        features['PREV-LABEL_' + str(j) + "START"] = 1
      else:
        features['PREV-LABEL_' + str(j) + str(previous_labels[j])] = 1
    for j in range(self.tag_depth):
      if j >= len(previous_tags):
        features['PREV-TAG_' + str(j) + "START"] = 1
      else:
        features['PREV-TAG_' + str(j) + previous_tags[j]] = 1
    if i < example.length - 1:
      features['NEXT-TAG_' + example.tags[i + 1]] = 1

    word = example.words[i].lower()
    tag = example.tags[i]

    features['WORD_' + word] = 1
    features['TAG_' + tag] = 1
    
    features['PHRASE-LENGTH'] = example.length
    #features['INDEX'] = i
    
    if self.tree_features:
      n = example.terminals[i]
      if n.parent:
        features['PARENT_' + n.parent.phrase] = 1
        #if n.parent.parent:
        #  features['GRAND_PARENT_' + n.parent.parent.phrase] = 1
      #for s in n.siblings:
      #  features['SIBLING_' + s.phrase] = features.get('SIBLING_' + s.phrase, 0) + 1
      
      features['ANCESTOR_NP'] = 'NP' in n.parent_phrases
      #features['ANCESTOR_VP'] = 'VP' in n.parent_phrases
      #features['ANCESTOR_ADJP'] = 'ADJP' in n.parent_phrases
      features['ANCESTOR_ADVP'] = 'ADVP' in n.parent_phrases

    for f in features.keys()[:]:
      for d in self.deleted_features:
        if d in f:
          del features[f]

    return features
  
  def train(self, examples):
    feature_list = []
    y = []
    for example in examples:
      for i in range(example.length):
        previous_labels = example.labels[max(i - self.label_depth, 0):i]
        feature_list.append(self.get_features(example, i, previous_labels))
        y.append(example.labels[i])
    
    X_train = self.vectorizer.fit_transform(feature_list)
    self.clf.fit(X_train, y)


class HMMPredictor(MarkovModel):
  def __init__(self, num_tags, label_depth=2, tag_depth=1):
    MarkovModel.__init__(self, num_tags, label_depth, tag_depth)
    self.history_to_label_counts = {}
    self.tag_to_label_counts = {}
    self.label_counts = Counter()

    self.transition_size = 0
    for depth in range(1 + max(label_depth, tag_depth)):
      depth_size = 1
      if depth <= label_depth: depth_size *= (3 ** depth)
      if depth <= tag_depth: depth_size *= (num_tags ** depth)
      self.transition_size += depth_size

  def get_log_scores(self, example, i, previous_labels):
    transition_smoothing = 0.0001
    emission_smoothing = 0.0001
    
    tag = example.tags[i]
    previous_tags = example.tags[max(i - self.tag_depth, 0):i]

    history = self.get_history(previous_tags, previous_labels) 
    transition = Counter()
    if history in self.history_to_label_counts:
      transition = self.history_to_label_counts[history]
    emission = self.tag_to_label_counts.get(tag, self.label_counts)
    
    transition_total = float(sum(transition.values())) + self.transition_size * transition_smoothing
    emission_total = float(sum(emission.values())) + self.num_tags * emission_smoothing

    log_scores = Counter()
    for label in range(3):
      p = ((transition_smoothing + transition[label]) / transition_total) \
        * ((emission_smoothing + emission[label]) / emission_total) \
        / self.label_counts[label]
      log_scores[label] = log(p)
    
    return log_scores
  
  def train(self, examples):
    for example in examples:
      for i in range(example.length):
        history = self.get_history(example.tags[max(i - self.tag_depth, 0):i], 
                                   example.labels[max(i - self.label_depth, 0):i])
        tag, label = example.tags[i], example.labels[i]
        
        self.history_to_label_counts.setdefault(history, Counter())[label] += 1
        self.tag_to_label_counts.setdefault(tag, Counter())[label] += 1
        self.label_counts[label] += 1

  def get_history(self, previous_tags, previous_labels):
    return " ".join(previous_tags) + \
           " ".join([str(label) for label in previous_labels])
