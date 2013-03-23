import json
import random
#import util

TEST_SIZE = 150
VALIDATION_SIZE = 150

class Example:
  def __init__(self, phrase, tagged, tree, labels):
    self.phrase = phrase
    self.tagged = tagged
    self.tree = tree
    self.labels = labels

class Loader:
  def __init__(self):
    phrases, tagged_phrases, trees, labels = [], [], [], []
    
    with open('../annotation/gold_labels.json') as f:
      gold = json.loads(f.read())
      for phrase_group in gold:
        for phrase, label in phrase_group:
          phrases.append(phrase)
          labels.append(label)
          trees.append("tr")

    with open('../preprocessed_data/tagged_phrases') as f:
      for tagged_phrase in f:
        tagged_phrases.append([tuple(word_tag.split("_")) for word_tag 
                               in tagged_phrase.strip().split(" ")])
    
    for tagged_phrase in tagged_phrases:
      for word_tag in tagged_phrase:
        assert(len(word_tag) == 2)
    assert(len(phrases) == len(tagged_phrases))
    assert(len(phrases) == len(trees))
    assert(len(phrases) == len(labels))

    self.examples = []
    for i in range(len(phrases)):
      self.examples.append(Example(phrases[i], tagged_phrases[i], 
                                   trees[i], labels[i]))
    
    # For a couple phrases, the stanford POS tagger and parser split up some 
    # words, making the tagged phrase not align with the labels. This removes
    # such phrases (22 of them)
    self.examples = [e for e in self.examples if len(e.phrase.split(" ")) == len(e.tagged)]

    new_indicies = range(len(self.examples))
    random.seed(0)
    random.shuffle(new_indicies)
    self.examples = [self.examples[new_indicies[i]] 
                     for i in range(len(self.examples))]

    self.num_examples = len(self.examples)
    num_train = self.num_examples - TEST_SIZE - VALIDATION_SIZE
    self.train = self.examples[0:num_train]
    self.validation = self.examples[num_train:num_train + VALIDATION_SIZE]
    self.test = self.examples[num_train + VALIDATION_SIZE:self.num_examples]
    #print len(self.train)
    #print len(self.test)
    #print len(self.validation)

l = Loader()
