class TreeBaselinePredictor:
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
