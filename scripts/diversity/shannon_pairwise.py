#!/usr/bin/python3
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
    parser.add_argument('-i', '--iterations', required=False, help='the number of iterations for the bootstrap.', default='1000')
    parser.add_argument('-s', '--sample-size', required=False, help='the sample size (in sentences).', default='100')

    args = parser.parse_args()

    lang = args.language

    sentences = {}
    nlps = {}
    lemmas={}

    metrics = {'SIMP':'compute_simpDiv', 'INVSIMP': 'compute_invSimpDiv', 'SHAN':'compute_shannonDiv'}
    metrics_bs = {}
    
    length = 0
 
    nlpD = spacy_udpipe.load(lang)
    nlpD.max_length = 300000000

    # 1. read all the file
    for textfile in args.files:
        system = os.path.splitext(os.path.basename(textfile))[0]
        sentences[system] = []
        
        with codecs.open(textfile, 'r', 'utf8') as ifh:
            tmp = [s.strip() for s in ifh.readlines()]
            sentences[system] = tmp
        
        if length == 0:
            length = len(sentences[system])

        lemmas[system] = get_lemmas(sentences[system], nlpD, system)
    # 2. Compute overall metrics
    for metric in metrics:
        print(metric)
        for syst in sentences:
            a = time.time() 
            print(syst, end=": ")
            score, _metr_dict = eval(metrics[metric])(lemmas[syst])
            print(str(score))

    sys.exit("Done (no significance)")
    
    # 4. read the other variables.
    iters = int(args.iterations)
    sample_size = int(args.sample_size)
    sample_idxs = np.random.randint(0, length, size=(iters, sample_size))

            
    for metric in metrics:
        metrics_bs[metric] = {}
        for syst in sentences:
            metrics_bs[metric][syst] = compute_shan_metric(metrics[metric], sentences[syst], sample_idxs, iters)

    for metric in metrics:
        print("-------------------------------------------------")
        print(metric)
        sign_scores = compute_significance(metrics_bs[metric], iters)
        print_latex_table(sign_scores, metric)
        sign_scores = compute_ttest_scikit(metrics_bs[metric], iters)
        print_latex_table(sign_scores, metric)
        
    simpDivD=compute_simpDiv(lemmaWordD)
    invSimpDiv=compute_invSimpDiv(lemmaWordD)
    shannonDiv=compute_shannonDiv(lemmaWordD)
      
    print(str(computeAverageValue(simpDivD)))
    print(str(computeAverageValue(invSimpDiv)))
    print(str(computeAverageValue(shannonDiv)))

if __name__=="__main__":
    main()
