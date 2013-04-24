class BaselinePredictor:
  def __init__(self, aspect_tags, value_tags):
    self.aspect_tags = aspect_tags
    self.value_tags = value_tags

  def train(self, examples):
    pass

  def predict(self, example):
    labels = []
    for i, (word, tag) in enumerate(example.tagged):
      labels.append(2)

      if any([aspect_tag in tag for aspect_tag in self.aspect_tags]):
        labels[i] = 0
      if any([value_tag in tag for value_tag in self.value_tags]):
        labels[i] = 1

    return labels

class TreeBaselinePredictor(BaselinePredictor):
  def __init__(self, aspect_tags, value_tags):
    BaselinePredictor.__init__(self, aspect_tags, value_tags)
  
    self.value_phrases = set(['NP', 'ADJP', 'ADVP'])
    self.aspect_phrases = set(['NP'])
    self.skip_tags = set(['"', ',', '-LRB-', '-RRB-', '.', ':', 'DT', 'CC'])

  def train(self, examples):
    pass

  def predict(self, example):
    labels = BaselinePredictor.predict(self, example)

    self.propagate_labels(example.tree, 0, self.aspect_phrases, self.skip_tags,
                          labels, example.tags)
    self.propagate_labels(example.tree, 1, self.value_phrases, self.skip_tags,
                          labels, example.tags)
    return labels

  def propagate_labels(self, n, label, phrase_types, skip_tags, labels, tags):
    def propagate_labels_recursive(n, i):
      if n.terminal:
        i[0] += 1
        return [i[0]]
      
      span = []
      for child in n.children:
        span += propagate_labels_recursive(child, i)
      
      span_lables = [labels[j] for j in span]
      if n.phrase in phrase_types and \
         all([l == label or l == 2 for l in span_lables]) and \
         label in span_lables:
        for j in span:
          if tags[j] not in skip_tags:
            labels[j] = label

      return span
    
    propagate_labels_recursive(n, [-1])

