This project investigates different approaches for extracting aspects and values in restaurant reviews. A complete description of the methods used is in writeup.pdf. This was done as a class project for the University of Washington's Natural Language Processing course ([CSE 517](http://courses.cs.washington.edu/courses/cse517/13wi/)).

Use "python tester.py" to run the project. It runs the experiments in the writeup
and outputs tables in a form that can be copy-pasted into latex. 
It requires python 2.7 and the scikit-learn python module to run.

### Complete description of files:
data contains the both the annotated and preprocessed data:
   - gold_labels.json contains the phrases with word label annotations
   - phrases contains the phrases, one per line
   - tagged_phrases contains the output of the Stanford POS Tagger on the
     phrases
   - parse_trees contains the output of the Stanford Parser on the phrases

code contains the project source code:
   - preprocesser.py runs the Stanford POS Tagger and Parser over the data. It
     is not necessary to run, since the preprocessed is already included in the
     data directory. If you want to run it anyway, it requires the Stanford
     POS Tagger and Parser.
   - loader.py reads the data in and does some basic processing, such as 
     deserializing the parse trees
   - rule_based_models.py contain the rule-based predictors
   - markov_models.py contains the HMM and MEMM predictors
   - tester.py runs the experiments in the report
