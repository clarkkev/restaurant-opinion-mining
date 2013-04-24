from rule_based_models import BaselinePredictor, TreeBaselinePredictor
from markov_models import HMMPredictor, MEMMPredictor
from tree_models import TreePredictor
from loader import Loader

def get_metrics(predictions, examples, target):
  tp = fp = fn = 0
  for prediction, example in zip(predictions, examples):
    #print "PHRASE: " + str(example.phrase)
    #print "TAGGED: " + str(example.tagged)
    #print "PREDIC: " + str(prediction)
    #print "ACTUAL: " + str(example.labels)
    #print
    for predicted, actual in zip(prediction, example.labels):
      if predicted == target:
        if predicted == actual:
          tp += 1
        else:
          fp += 1
      elif actual == target:
         fn += 1
  
  p = float(tp) / (tp + fp)
  r = float(tp) / (tp + fn)
  f1 = (2 * p * r) / (p + r)
  print f1

  return [100 * p, 100 * r, 100* f1]

def output_table(rows):
  table = """
\\begin{tabular}{| l | c  c  c | c  c  c |}
\\hline
   & \\multicolumn{3}{| c |}{Aspect} & \\multicolumn{3}{| c |}{Value} \\\\
   & \\multicolumn{1}{c}{Precision} & \\multicolumn{1}{c}{Recall} & \\multicolumn{1}{c |}{F$_1$} 
   & \\multicolumn{1}{c}{Precision} & \\multicolumn{1}{c}{Recall} & \\multicolumn{1}{c |}{F$_1$} \\\\ \\hline
"""
  for row in rows:
    table += "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline\n".format(*row)
  table += "\\end{tabular}\n"
  
  print table

def evaluate(predictor, name, train, test):
  predictor.train(train)
  predictions = [predictor.predict(example) for example in test]
  return [name] + get_metrics(predictions, test, 0) + get_metrics(predictions, test, 1)

if __name__ == '__main__':
  loader = Loader()

  '''print "TESTING RULE-BASED MODELS ON VALIDATION SET:"
  table1 = []
  table1.append(evaluate(BaselinePredictor(['NN'], ['JJ']), 
    "TAG-SMALL", loader.train, loader.validation))
  table1.append(evaluate(BaselinePredictor(['NN'], ['CD', 'RB', 'JJ', 'VBG', 'VBN']),
    "TAG-LARGE", loader.train, loader.validation))
  table1.append(evaluate(TreeBaselinePredictor(['NN'], ['JJ']),
    "TREE-SMALL", loader.train, loader.validation))
  table1.append(evaluate(TreeBaselinePredictor(['NN'], ['CD', 'RB', 'JJ', 'VBG', 'VBN']),
    "TREE-LARGE", loader.train, loader.validation))
  output_table(table1)

  print "FINDING BEST TAG DEPTH AND LABEL DEPTH FOR HMM ON VALIDATION SET:"
  table2 = []
  for tag_depth in range(3):
    for label_depth in range(3):
      hmm = HMMPredictor(loader.num_tags, label_depth, tag_depth)
      table2.append(evaluate(hmm, "$d_t = " + str(tag_depth) + ", d_l = " + str(label_depth) + "$", 
                    loader.train, loader.validation))
  output_table(table2)

  print "FINDING BEST TAG DEPTH AND LABEL DEPTH FOR MEMM ON VALIDATION SET:"
  table3 = []
  for tag_depth in range(1, 4):
    for label_depth in range(1, 4):
      hmm = MEMMPredictor(loader.num_tags, label_depth, tag_depth)
      table3.append(evaluate(hmm, "$d_t = " + str(tag_depth) + ", d_l = " + str(label_depth) + "$", 
                    loader.train, loader.validation))
  output_table(table3)

  print "RUNNING ABLATION STUDY ON MEMM FEATURES ON VALIDATION SET:"
  table4 = []
  ablations = ['NONE-DELETED', 'NEXT-TAG', 'WORD', 'PHRASE-LENGTH']
  for feature in ablations:
    memmm = MEMMPredictor(loader.num_tags, 3, 1, [feature])
    table4.append(evaluate(memmm, feature, 
                  loader.train, loader.validation))
  output_table(table4)
  
  print "RUNNING ABLATION STUDY ON TREE MEMM FEATURES ON VALIDATION SET:"
  table5 = []
  ablations = ["NONE-DELETED", "PARENT", "ANCESTOR_NP", "ANCESTOR_ADVP"]
  for feature in ablations:
    memmm = MEMMPredictor(loader.num_tags, 3, 1, [feature], tree_features=True)
    table5.append(evaluate(memmm, feature, 
                  loader.train, loader.validation))
  output_table(table5)

  print "FINAL SCORES ON TEST SET:"
  ruled = BaselinePredictor(['NN'], ['CD', 'RB', 'JJ', 'VBG', 'VBN'])
  hmm = HMMPredictor(loader.num_tags, 1, 1)
  memm = MEMMPredictor(loader.num_tags, 3, 1)
  memm_tree = MEMMPredictor(loader.num_tags, 3, 1, tree_features=True)
  table6 = []
  table6.append(evaluate(ruled, "TAG-LARGE",
                loader.train, loader.test))
  table6.append(evaluate(hmm, "HMM ($d_t = 1, d_l = 1$)",
                loader.train, loader.test))
  table6.append(evaluate(memm, "MEMM ($d_t = 1, d_l = 3$)",
                loader.train, loader.test))
  table6.append(evaluate(memm_tree, "MEMM-TREE ($d_t = 1, d_l = 3$)",
                loader.train, loader.test))
  output_table(table6)'''


memm = TreePredictor(loader.num_tags)#MEMMPredictor(loader.num_tags, 0, 1, tree_features=True)
evaluate(memm, "MEMM", loader.train, loader.validation)
