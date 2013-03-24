import json
import re
import random

TEST_SIZE = 150
VALIDATION_SIZE = 150

class PennNode:
  current_nid = 0
  phrase_p = re.compile('[(](\S+)\s*(.*)')
  preterm_p = re.compile('[(]([^()\s]*)\s([^()\s]*)[)]')

  def __init__(self, depth, contents, parent=None):
    self.parent = parent
    self.children = []
    self.depth = depth
    self.terminal = None
    self.nid = PennNode.current_nid
    PennNode.current_nid += 1

    if PennNode.preterm_p.match(contents):
      preterm_matches = PennNode.preterm_p.findall(contents)
      if len(preterm_matches) == 1:
        preterm_match = preterm_matches[0]
        self.phrase = preterm_match[0]
        self.terminal = preterm_match[1]
      else:
        preterm_match = preterm_matches[0]
        self.phrase = preterm_match[0]
        self.terminal = preterm_match[1]
        for phrase, terminal in preterm_matches[1:]:
          PennNode(depth, "(" + phrase + " " + terminal + ")", parent)

    else:
      phrase_match = PennNode.phrase_p.match(contents)
      self.phrase = phrase_match.group(1)
      for preterm_match in PennNode.preterm_p.finditer(phrase_match.group(2)):
        PennNode(depth + 1, preterm_match.group(0), self)

    if parent:
      parent.children.append(self)

  def display(self):
    display_str = ' ' * self.depth + '(' + self.phrase
    if self.terminal:
      display_str += ' ' + self.terminal + ')'
      print display_str
      return

    nonterminal_seen = False
    for n in self.children:
      if n.terminal and not nonterminal_seen:
        display_str += ' (' + n.phrase + ' ' + n.terminal + ')'
      else:
        if not nonterminal_seen:
          nonterminal_seen = True
          print display_str
        n.display()
    if not nonterminal_seen:
      print display_str

def buildTree(lines):
  root = n = None
  tokens = []

  for line in lines:
    depth = 0
    while line[depth] == ' ':
      depth += 1

    line = line.strip()
    if depth == 0:
      root = n = PennNode(0, line)
    else:
      while depth/2 <= n.depth:
        n = n.parent
      n = PennNode(depth/2, line, n)
  return root

def terminals_from_tree(n):
  if n.terminal:
    return [n]
  tokens = []
  for child in n.children:
    tokens += terminals_from_tree(child)
  return tokens

class Example:
  def __init__(self, phrase, tagged, tree, labels):
    self.phrase = phrase
    self.tagged = tagged
    self.words, self.tags = zip(*self.tagged)
    self.tree = tree
    self.labels = labels
    self.length = len(labels)

class Loader:
  def __init__(self):
    phrases, tagged_phrases, trees, labels = [], [], [], []

    with open('../data/gold_labels.json') as f:
      gold = json.loads(f.read())
      for phrase_group in gold:
        for phrase, label in phrase_group:
          phrases.append(phrase)
          labels.append(label)

    with open('../data/tagged_phrases') as f:
      for tagged_phrase in f:
        tagged_phrases.append([tuple(word_tag.split("_")) for word_tag
                               in tagged_phrase.strip().split(" ")])

    with open('../data/parse_trees') as f:
      tree_lines = []
      for line in f:
        line = line.rstrip()
        if len(line) == 0:
          trees.append(buildTree(tree_lines))
          tree_lines = []
        else:
          tree_lines.append(line)

    for tagged_phrase in tagged_phrases:
      for word_tag in tagged_phrase:
        assert(len(word_tag) == 2)
    assert(len(phrases) == len(tagged_phrases))
    assert(len(phrases) == len(trees))
    assert(len(phrases) == len(labels))

    self.tagset = set()
    self.examples = []
    for i in range(len(phrases)):
      example = Example(phrases[i], tagged_phrases[i],
                        trees[i], labels[i])
      self.examples.append(example)
      
      self.tagset |= set(example.tags)
      
      #terminals = terminals_from_tree(example.tree)
      #for i, label in enumerate(example.labels):
      #  terminals[i].terminal += "_" + str(label)

      #e = self.examples[i]
      #print e.phrase
      #print e.tree.tokens
      #e.tree.display()
      #print
    self.num_tags = len(self.tagset)

    
    self.examples = [e for e in self.examples if len(e.phrase.split(" ")) == len(e.tagged) and          
                                                 len(e.tagged) == len(terminals_from_tree(e.tree))]
    print "LOADED: " + str(len(self.examples)) +  " PHRASES"

    new_indicies = range(len(self.examples))
    random.seed(1)
    random.shuffle(new_indicies)
    self.examples = [self.examples[new_indicies[i]]
                     for i in range(len(self.examples))]


    self.num_examples = len(self.examples)
    num_train = self.num_examples - TEST_SIZE - VALIDATION_SIZE
    self.train = self.examples[0:num_train]
    self.validation = self.examples[num_train:num_train + VALIDATION_SIZE]
    self.test = self.examples[num_train + VALIDATION_SIZE:self.num_examples]

l = Loader()
