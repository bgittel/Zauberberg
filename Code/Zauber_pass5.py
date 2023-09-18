from datetime import datetime
from faulthandler import disable
# from interruptingcow import timeout
# from joblib import Parallel, delayed
from logging import exception


import argparse
import ast
import csv
import json
import multiprocessing as mp
import re
import socket
import sys
import time
import traceback


import os
import spacy
import sys
import numpy as np
import pandas as pd
from statistics import mean

# import pipeline components
# notebook_path = os.getcwd()
# print(notebook_path)
sys.path.append("/home/bgittel/zauberberg/pipy-public/")  ### hier ggf. anpassen
from pipeline.components.analyzer import demorphy_analyzer
from pipeline.components.clausizer import dependency_clausizer
from pipeline.components.lemma_fixer import lemma_fixer
from pipeline.components.sentencizer import spacy_sentencizer
from pipeline.components.tense_tagger import rb_tense_tagger
from pipeline.components.reflection_tagger import neural_reflection_tagger


# import TextGrid components
from tgclients.aggregator import Aggregator
aggregator = Aggregator()
import time


from settings import PARSING_PATH

def df_line(text) -> dict:
    
    t1 = time.time()

    # build the pipeline
    nlp = spacy.load(os.path.join(PARSING_PATH, "de_ud_lg"))
    nlp.add_pipe(spacy_sentencizer, name="sentencizer", before="parser")
    nlp.add_pipe(lemma_fixer, name="lemma_fixer", after="tagger")
    nlp.add_pipe(demorphy_analyzer, name="analyzer")
    nlp.add_pipe(dependency_clausizer, name="clausizer")
    nlp.add_pipe(rb_tense_tagger, name="tense_tagger")
    nlp.add_pipe(neural_reflection_tagger, name="reflection_tagger")
    nlp.max_length = 4000001
    print('nlp.max_length is:', nlp.max_length)
    print(nlp.pipe_names)
    doc = nlp(text)

    #calculate reflection scores: append `reflective_clauses_scored` and `reflection_score_mean`
    reflection_scores = list()
    reflective_clauses_new = 0.0
    reflective_clauses_scored = 0.0
    for clause in doc._.clauses:
        labels = set()
        if (clause._.rps is not None) and len(clause._.rps) > 0:
            for rp in clause._.rps:
                labels.update(rp.tags)
        gi, nfr, comment = 0.0, 0.0, 0.0
        if 'RP' in labels:
            print("Classifier used in binary mode, use multi-label mode!")
            exit()
        if 'GI' in labels or "Nichtfiktional" in labels or 'Comment' in labels:
            reflective_clauses_new += 1
        if 'GI' in labels:
            gi = 1.0
        if "Nichtfiktional" in labels:
            nfr = 1.0
        if 'Comment' in labels:
            comment = 1.0
        reflection_score = calculate_reflection_score(gi=gi, nfr=nfr, comment=comment)
        if reflection_score >= 0.5:
            reflective_clauses_scored += 1
        reflection_scores.append(reflection_score)
    reflection_score_mean = mean(reflection_scores)

    t2 = time.time()
    print("\n\n")
    return [{
                    "text_length_sents": len(list(doc.sents)),
                    "text_length_clauses": len(doc._.clauses),
                    "text_length_tokens": len(doc),
                    "reflective_passages_global": len(doc._.rps),
                    "reflective_clauses": len(set([clause for passage in doc._.rps for clause in passage.clauses])),
                    "reflective_tokens": len(set([token for passage in doc._.rps for token in passage.tokens])),
                    "gi_passages_global": len(set([passage for passage in doc._.rps if "GI" in passage.tags])),
                    "gi_clauses": len(set([clause for passage in doc._.rps if "GI" in passage.tags for clause in passage.clauses])),
                    "gi_tokens": len(set([token for passage in doc._.rps if "GI" in passage.tags for token in passage.tokens])),
                    "nfr_passages_global": len(set([passage for passage in doc._.rps if "Nichtfiktional" in passage.tags])),
                    "nfr_clauses": len(set([clause for passage in doc._.rps if "Nichtfiktional" in passage.tags for clause in passage.clauses])),
                    "nfr_tokens": len(set([token for passage in doc._.rps if "Nichtfiktional" in passage.tags for token in passage.tokens])),
                    "comment_passages_global": len(set([passage for passage in doc._.rps if "Comment" in passage.tags])),
                    "comment_clauses": len(set([clause for passage in doc._.rps if "Comment" in passage.tags for clause in passage.clauses])),
                    "comment_tokens": len(set([token for passage in doc._.rps if "Comment" in passage.tags for token in passage.tokens])),
                    "comment_nfr_passages_global": len(set([passage for passage in doc._.rps if "Comment" in passage.tags and "Nichtfiktional" in passage.tags])),
                    "comment_nfr_clauses": len(set([clause for passage in doc._.rps if "Comment" in passage.tags and "Nichtfiktional" in passage.tags for clause in passage.clauses])),
                    "comment_nfr_tokens": len(set([token for passage in doc._.rps if "Comment" in passage.tags and "Nichtfiktional" in passage.tags for token in passage.tokens])),
                    "nfr_gi_passages_global": len(set([passage for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags])),
                    "nfr_gi_clauses": len(set([clause for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags for clause in passage.clauses])),
                    "nfr_gi_tokens": len(set([token for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags for token in passage.tokens])),
                    "comment_gi_passages_global": len(set([passage for passage in doc._.rps if "Comment" in passage.tags and "GI" in passage.tags])),
                    "comment_gi_clauses": len(set([clause for passage in doc._.rps if "Comment" in passage.tags and "GI" in passage.tags for clause in passage.clauses])),
                    "comment_gi_tokens": len(set([token for passage in doc._.rps if "Comment" in passage.tags and "GI" in passage.tags for token in passage.tokens])),
                    "comment_gi_nfr_passages_global": len(set([passage for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags and "Comment" in passage.tags])),
                    "comment_gi_nfr_clauses": len(set([clause for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags and "Comment" in passage.tags for clause in passage.clauses])),
                    "comment_gi_nfr_tokens": len(set([token for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags and "Comment" in passage.tags for token in passage.tokens])),
                    "reflection_score_mean": reflection_score_mean,
                    "reflective_clauses_scored": reflective_clauses_scored,
                    "reflective_clauses_new": reflective_clauses_new,
                    "exception": "",
                    "setting": ""
                    },
                {
                    "nfr_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Nichtfiktional" in passage.tags for clause in passage.clauses])
                        ),
                    "reflective_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps for clause in passage.clauses])
                        ),
                    "gi_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "GI" in passage.tags for clause in passage.clauses])
                        ),
                    "comment_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Comment" in passage.tags for clause in passage.clauses])
                        ),
                    "gi_nfr_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags for clause in passage.clauses])
                        ),
                    "comment_gi_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Comment" in passage.tags and "GI" in passage.tags for clause in passage.clauses])
                        ),
                    "comment_nfr_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Comment" in passage.tags and "Nichtfiktional" in passage.tags for clause in passage.clauses])
                        ),
                    "comment_gi_nfr_clauses": ListMaker(
                        set([clause.text for passage in doc._.rps if "Nichtfiktional" in passage.tags and "GI" in passage.tags and "Comment" in passage.tags for clause in passage.clauses])
                        )

                    }
            ]

def arg_to_bool(arg):
    if not isinstance(arg, str):
        return arg
    elif arg.lower() in ["true", "1", "yes"]:
        return True
    elif arg.lower() in ["false", "0", "no"]:
        return False
    else:
        return False

def sigmoid(x):
    ex = np.exp(x)
    return ex / (1 + ex)

def calculate_reflection_score(gi: float, nfr: float, comment: float) -> float:
    return sigmoid(0.7155*gi + 0.3398*nfr + 1.2862*comment - 0.6147*(gi*comment) - 0.7238)

def ListMaker(set_obj):
    output = []
    for x in set_obj:
        output.append(x)
    return output
    

text_kafka_verwandlung = aggregator.text("textgrid:qn07.0").text
# !jupyter nbextension enable --py widgetsnbextension
# text_file = open("Zauberberg.txt")
# #read whole file to a string
# data = text_file.read()
# #close file
# text_file.close()
#print("...")
#print( df_line(text_kafka_verwandlung) )