#!/home/dimitar/anaconda3/envs/lex_div/bin/python
# -*- coding: utf-8 -*-

import codecs
import statistics
import argparse
import os
import numpy as np
from scipy.stats import ttest_ind
from mosestokenizer import *
from biasmt_metrics import *
import sys
import time

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Extracts words to a dictionary with their frequencies.')
    parser.add_argument('-f', '--files', required=True, help='the files to read.', nargs='+')
    parser.add_argument('-l', '--language', required=False, help='the language.', default='en')
    parser.add_argument('-v', '--frequency-vocabulary', required=False, help='a frequency vocabulary', default=None)
    parser.add_argument('-t', '--top-words', required=False, help='number of words with top frequencies.', default='1000')

    parser.add_argument('-i', '--iterations', required=False, help='the number of iterations for the bootstrap.', default='1000')
    parser.add_argument('-s', '--sample-size', required=False, help='the sample size (in sentences).', default='100')

    args = parser.parse_args()

    lang = args.language

    freq_dict = []
    if args.frequency_vocabulary is not None:
        top = int(args.top_words)
        with open(args.frequency_vocabulary, "r") as iF:
            freq_dict = [line.strip().split()[0] for line in iF.readlines()[:top]]
    else:
        freq_dict = None

    sentences = {}

    metrics_bs = {}
    
    # 1. read all the file
    for textfile in args.files:
        system = os.path.splitext(os.path.basename(textfile))[0]
        sentences[system] = []
        
        with codecs.open(textfile, 'r', 'utf8') as ifh:
            sentences[system] = [s.strip() for s in ifh.readlines() if s.strip()] # ! Spacy UDPIPE crashes if we keep also empty lines

    # 2. Compute overall metrics
    for syst in sentences:
        print(syst, end=": ")
        score = compute_gram_diversity(sentences[syst], lang, syst, freq_dict)
        print(" & ".join([str(s) for s in score]))

    sys.exit("Done")
    
if __name__=="__main__":
    main()
