import json
import subprocess

PARSER_DIR = "~/Programming/tools/stanford-parser-2012-11-12"
TAGGER_DIR = "~/Programming/tools/stanford-postagger-2012-11-11/"
PHRASES_FILE = "../data/phrases"

gold = []
with open('../data/gold_labels.json') as f:
  gold = json.loads(f.read())

with open(PHRASES_FILE, 'w') as f:
  for phrase_group in gold:
    for phrase, label in phrase_group:
      f.write(phrase + '\n')

subprocess.call("java -mx300m -cp " + TAGGER_DIR + "stanford-postagger-3.1.4.jar" + \
  ": edu.stanford.nlp.tagger.maxent.MaxentTagger -sentenceDelimiter newline " + \
  "-model " + TAGGER_DIR + "models/english-left3words-distsim.tagger -textFile " + \
  PHRASES_FILE + " -outputFile ../data/tagged_phrases ", 
  shell=True)

subprocess.call("java -mx1024m -cp " + PARSER_DIR + \
  "/*: edu.stanford.nlp.parser.lexparser.LexicalizedParser " + \
  "-outputFormat penn -sentences newline " + \
  "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz " + \
  "> ../data/parse_trees " + \
  PHRASES_FILE, shell=True)
