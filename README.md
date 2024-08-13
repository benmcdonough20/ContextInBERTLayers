# Context encoding in layers of BERT
Final project for LING 227

Ben McDonough and Daniel Shimberg

May 2024

## Abstract
The introduction of pre-trained bi-directional
text encoders advanced the field of NLP by
creating a method for generating encodings car-
rying more context than previous sequential
models. In this work, we examine the contex-
tual information encoded in the hidden states
of BERT. We use a linear classifier trained on
hidden states from each layer of BERT to deter-
mine how the level of encoded context changes
between layers. Our main findings are that
BERT captures the dependence between the
subject and main verb of a sentence at early lay-
ers, and captures the extralinguistic dependence
of a pronoun on its referent between sentences
at a later stage. We hypothesize that contextual
encoding is also largely effected by the textual
proximity of two words. We provide evidence
that this is the case by comparing the impact of
gender and proximity on pronoun reference.

[writeup](https://www.overleaf.com/project/662c108c86362e3fa8e82dbb)


## Code and Documentation

We ran three experiments, each of which is contained in a separate folder:
- Subject - Verb agreement
- Subject - Pronoun agreement
- Gender agreement

The base case was similar enough to Experiment 1 that we do not include it as additional code.
Each of the three folders contains similar files, `CFG.py` and `BERT_classifier.py` with extra helper methods where necessary,
particularly in the CFG generator for the gender agreement experiment. The most robust comments can 
be found in Experiment 1.

## Running

To run any experiment, navigate to the requisite folder and run
1. `python CFG.py`
2.  `python BERT_classifier.py`
   
or replace these with the appropriate file name where necessary.
