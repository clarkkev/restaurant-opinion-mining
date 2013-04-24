import json
import re
import random

TEST_SIZE = 150
VALIDATION_SIZE = 150

class Node:
  current_nid = 0
  phrase_p = re.compile('[(](\S+)\s*(.*)')
  preterm_p = re.compile('[(]([^()\s]*)\s([^()\s]*)[)]')

  def __init__(self, depth, contents, parent=None):
    self.parent = parent
    self.children = []
    self.depth = depth
    self.height = 0
    self.terminal = None
    self.nid = Node.current_nid
    self.siblings = []
    self.parent_phrases = set()
    Node.current_nid += 1

    if Node.preterm_p.match(contents):
      preterm_matches = Node.preterm_p.findall(contents)
      if len(preterm_matches) == 1:
        preterm_match = preterm_matches[0]
        self.phrase = preterm_match[0]
        self.terminal = preterm_match[1]
      else:
        preterm_match = preterm_matches[0]
        self.phrase = preterm_match[0]
        self.terminal = preterm_match[1]
        for phrase, terminal in preterm_matches[1:]:
          Node(depth, "(" + phrase + " " + terminal + ")", parent)

    else:
      phrase_match = Node.phrase_p.match(contents)
      self.phrase = phrase_match.group(1)
      for preterm_match in Node.preterm_p.finditer(phrase_match.group(2)):
        Node(depth + 1, preterm_match.group(0), self)

    if parent:
      parent.children.append(self)

  def set_features(self):
    self.siblings = []
    for n in self.children:
      n.set_features()
    if self.parent:
      for n in self.parent.children:
        self.siblings.append(n)
    
    self.parent_phrases = set()
    n = self.parent
    while n:
      self.parent_phrases.add(n.phrase)
      n = n.parent

  def set_height(self):
    if self.terminal:
      self.height = 0
    else:
      self.height = 1 + max(child.set_height() for child in self.children)
    return self.height

  def set_span(self, i=[0]):
    if self.terminal:
      self.span = [i[0]]
      i[0] += 1
    else:
      self.span = []
      for child in self.children:
        self.span += child.set_span(i)
    return self.span

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
      root = n = Node(0, line)
    else:
      while depth/2 <= n.depth:
        n = n.parent
      n = Node(depth/2, line, n)
  return root

def terminals_from_tree(n):
  if n.terminal:
    return [n]
  tokens = []
  for child in n.children:
    tokens += terminals_from_tree(child)
  return tokens

class Example:
  def __init__(self, phrase, tagged, tree, labels, terminals):
    self.phrase = phrase
    self.tagged = tagged
    self.words, self.tags = zip(*self.tagged)
    self.tree = tree
    self.labels = labels
    self.length = len(labels)
    self.terminals = terminals

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
      trees[i].set_features()
      trees[i].set_span([0])
      trees[i].set_height()
      example = Example(phrases[i], tagged_phrases[i],
                        trees[i], labels[i], terminals_from_tree(trees[i]))
      
      #print trees[i].span, example.length
      self.examples.append(example)
      self.tagset |= set(example.tags)
      
    self.num_tags = len(self.tagset)
    self.examples = [e for e in self.examples if len(e.phrase.split(" ")) == len(e.tagged) and          
                                                 len(e.tagged) == len(terminals_from_tree(e.tree))]
    print "LOADED: " + str(len(self.examples)) +  " PHRASES,"
    print str(sum(example.length for example in self.examples)) + " WORDS"

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

if __name__ == '__main__':
  l = Loader()
