import json
import re
import random
#import util

TEST_SIZE = 150
VALIDATION_SIZE = 150

class Tree:
  def __init__(self, lines):
    self.root = n = None
    self.tokens = []
    
    for line in lines:
      depth = 0
      while line[depth] == ' ':
        depth += 1
      
      line = line.strip()
      if depth == 0:
        self.root = n = Node(0, line)
      else:
        while depth/2 <= n.depth:
          n = n.parent
        n = Node(depth/2, line, n)
    
    def get_tokens(n):
      if n.terminal:
        return [(n.terminal, n.phrase)]
      tokens = []
      for child in n.children:
        tokens += get_tokens(child)
      return tokens
    self.tokens = get_tokens(self.root)

  def display(self):
    self.root.display()


class Node:
  current_nid = 0
  phrase_p = re.compile('[(](\S+)\s*(.*)')
  preterm_p = re.compile('[(]([^()\s]*)\s([^()\s]*)[)]')

  def __init__(self, depth, contents, parent=None):
    self.parent = parent
    self.children = []
    self.depth = depth
    self.terminal = None
    self.nid = Node.current_nid
    Node.current_nid += 1

    # a bit hacky here
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

    with open('../preprocessed_data/tagged_phrases') as f:
      for tagged_phrase in f:
        tagged_phrases.append([tuple(word_tag.split("_")) for word_tag 
                               in tagged_phrase.strip().split(" ")])
 
    with open('../preprocessed_data/parse_trees') as f:
      tree_lines = []
      for line in f:
        line = line.rstrip()
        if len(line) == 0:
          trees.append(Tree(tree_lines))
          tree_lines = []
        else:
          tree_lines.append(line)

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
      #e = self.examples[i]
      #print e.phrase
      #print e.tree.tokens
      #e.tree.display()
      #print   
    
    # For a couple phrases, the stanford POS tagger and parser split up some 
    # words, making the tagged phrase not align with the labels. This removes
    # such phrases (22 of them)
    self.examples = [e for e in self.examples if len(e.phrase.split(" ")) == len(e.tagged)]
    for e in self.examples:
      if(len(e.tagged) != len(e.tree.tokens)):
        print e.tagged
        print e.tree.tokens

      assert(len(e.tagged) == len(e.tree.tokens))

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

l = Loader()
