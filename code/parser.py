from loader import terminals_from_tree, Loader
import random

NONTERMINAL_ANNOTATIONS = 20

class RuleInfo():
  def __init__(self, tagset):
    get_annotations = lambda label, n: [label + str(i) for i in range(n)]
    
    self.label_to_annotations = {tag: get_annotations(tag, 3) for tag in tagset}
    self.label_to_annotations['Z'] = get_annotations('Z', NONTERMINAL_ANNOTATIONS)
    self.all_annotations = reduce(lambda l1, l2: l1 + l2, self.label_to_annotations.values())

class ParserModel():
  def train(self, examples, tagset):
    self.trees = get_trees(examples)
    r = RuleInfo(tagset)

class Tree:
  def __init__(self, children, label='Z'):
    self.children = children
    self.label = label
    self.terminal = (label != 'Z')

    self.annotations = 3 if self.terminal else NONTERMINAL_ANNOTATIONS
    self.zs = [0] * self.annotations
    
    self.inner = [0] * self.annotations
    self.outer = [0] * self.annotations

  def reset_probs(self):
    self.inner = self.outer = {}
    for child in children:
      child.reset_probs()

  def initialize_zs(self):
    if self.terminal:
      zs[label] = 1
    else:
      self.zs = [1 + random.random() * 0.2 for i in range(self.annotations)]
      total = sum(self.zs)
      self.zs = map(lambda p: p / total, self.zs)
      for child in children:
        child.initialize_zs()


def collapse_unaries(tree):
  #while len(tree.children) == 1 and len(tree.children[0].children) == 1:
  #  tree.children[0].children = tree.children[0].children[0].children
  while len(tree.children) == 1:
    child = tree.children[0]
    if child.terminal:
      tree.terminal = child.terminal
    tree.children = child.children
  for child in tree.children:
    collapse_unaries(child)

def binarize_tree(n):
  if n.terminal:
    return Tree([], n.phrase)
  if len(n.children) == 1:
    return Tree([binarize_tree(n.children[0])])
  return binarize_tree_helper(n, 0)

def binarize_tree_helper(n, children_generated):
  left_tree = n.children[children_generated]
  children = []
  children.append(binarize_tree(left_tree))
  if children_generated < len(n.children) - 1:
    right_tree = binarize_tree_helper(n, children_generated + 1)
    children.append(right_tree)
  return Tree(children)

def get_trees(examples):
  trees = []
  for example in examples:
    tree = binarize_tree(example.tree)
    collapse_unaries(tree)
    terminals = terminals_from_tree(tree)
    for i, label in enumerate(example.labels):
      terminals[i].label = label

    assert(len(terminals) == len(terminals_from_tree(example.tree)))

    trees.append(tree)
  return trees

if __name__ == '__main__':
  loader = Loader()
  model = ParserModel()
  model.train(loader.train, loader.tagset)
