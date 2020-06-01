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
    parser.add_argument('-i', '--iterations', required=False, help='the number of iterations for the bootstrap.', default='1000')
    parser.add_argument('-s', '--sample-size', required=False, help='the sample size (in sentences).', default='100')

    args = parser.parse_args()

    lang = args.language

    sentences = {}

    metrics = {'GramDiv':'compute_gram_diversity'}
    metrics_bs = {}
    
    # 1. read all the file
    for textfile in args.files:
        system = os.path.splitext(os.path.basename(textfile))[0]
        sentences[system] = []
        
        with codecs.open(textfile, 'r', 'utf8') as ifh:
            sentences[system] = [s.strip() for s in ifh.readlines() if s.strip()] # ! Spacy UDPIPE crashes if we keep also empty lines

    # 2. Compute overall metrics
    for metric in metrics:
        print(metric)
        for syst in sentences:
            a = time.time()
            print(syst, end=": ")
            score = eval(metrics[metric])(sentences[syst], lang, syst)
            print(" & ".join([str(s) for s in score]))

    sys.exit("Done")
    
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
