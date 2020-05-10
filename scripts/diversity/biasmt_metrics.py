import itertools
from lexical_diversity import lex_div as ld
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import statistics
import spacy_udpipe
import time
import pickle
import os

def get_lemmas(sentences, nlpD, system_name):
    ''' Computes the lemmas and their frequencies for the given sentences
    
        :params sentences: a list of sentences
        :params nlpd: the data model for the lematizer
        :returns: a dictionary of lemmas and frequencies
    '''
    a = time.time()
    if os.path.exists(system_name + ".spacy_udpipe.model"):
        with open(system_name + ".spacy_udpipe.model", "rb") as SpUpM:
            nlps = pickle.load(SpUpM)
        print("Model loaded from file")
    else:
        nlps = nlpD(" ".join(sentences))
        with open(system_name + ".spacy_udpipe.model", "wb") as SpUpM:
            pickle.dump(nlps, SpUpM)
        print("Model built from scratch")
        
    lemmas = {}
    for token in nlps:
        lemma=token.lemma_    
        tokenLow=str(token).lower()

        if lemma in lemmas: # existing lemma
            if tokenLow not in lemmas[lemma]: 
                lemmas[lemma][tokenLow]=1
            else:
                lemmas[lemma][tokenLow]+=1
        else:                       # unexisting lemma
            lemmas[lemma]={}        # if this is the first time we have a lemma then there are no tokens
            lemmas[lemma][tokenLow]=1

    return lemmas
    

def simpson_diversity(wordFormDict):
    ''' Computes the Simpson Diversity Index
    
        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number) 
    '''

    def p(n, N):
        ''' Relative abundance 
        '''
        if n ==  0:
            return 0
        else:
            return float(n)/N

    N = sum(wordFormDict.values())
    return sum(p(n, N)**2 for n in wordFormDict.values() if n != 0)

def inverse_simpson_diversity(wordFormDict):
    ''' Computes the inverse Simpson Diversity Index
    
        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number) 
    '''
    return float(1)/simpson_diversity(wordFormDict)

"""# Shannon Diversity #
The Shannon-Weiner diversity represent the proportion of species abundance in the population. Its being at maximum when all species occur in similar number of individuals and the lowest when the sample contain one species. From my experience there is no limit to compare the diversity value with as for evenness, which resricted between 0-1. For Example, if the sample contain 4 species each represented by 5o individuals the, diversity H equal 1.3863, and if the sample contain 5 species (one more) and each represented by similar number of individuals (50), the diversity equal 1.6094.
"""

def shannon_diversity(wordFormDict):
    '''
    
        :param wordFormDict: a dictionary { 'species': count }
        :returns: Shannon Diversity Index
    '''
    #>>> sdi({'a': 10, 'b': 20, 'c': 30,})
    #1.0114042647073518"""
    
    from math import log as ln
    
    def p(n, N):
        """ Relative abundance """
        if n ==  0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)
            
    N = sum(wordFormDict.values())
    
    return -sum(p(n, N) for n in wordFormDict.values() if n != 0)

def compute_simpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a nested dictionary
        :returns: a dictionary with the simpson diversity for every lemma 
    '''
    simpsonDict = {}
    for l in nestedDict:
        simpsonDict[l]=simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values()), simpsonDict

def compute_invSimpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
    
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the inversed simpson diversity
    '''
    simpsonDict={}
    for l in nestedDict:
        simpsonDict[l]=inverse_simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values()), simpsonDict 

def compute_shannonDiv(nestedDict):
    ''' Computes the shannon diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the shannon diversity
    '''
    shannonDict={}
    for lem in nestedDict:
        shannonDict[lem]=shannon_diversity(nestedDict[lem])
    return statistics.mean(shannonDict.values()), shannonDict

def compute_yules_i(sentences):
    ''' Computing Yules I measure

        :param sentences: dictionary with all words and their frequencies
        :returns: Yules I (the inverse of yule's K measure) (float) - the higher the better
    '''
    _total, vocabulary = get_vocabulary(sentences)
    M1 = float(len(vocabulary))
    M2 = sum([len(list(g))*(freq**2) for freq,g in itertools.groupby(sorted(vocabulary.values()))])

    try:
        return (M1*M1)/(M2-M1)
    except ZeroDivisionError:
        return 0

def compute_ttr(sentences):
    ''' Computes the type token ratio
    
        :param sentences: the sentences
        :returns: The type token ratio (float)
    '''      

    total, vocabulary = get_vocabulary(sentences)    
    return len(vocabulary)/total
    
def compute_mtld(sentences):
    ''' Computes the MTLD
    
        :param sentences: sentences
    
        :returns: The MTLD (float)
    '''      
    
    ll = ' '.join(sentences)
    return ld.mtld(ll)
    
def get_vocabulary(sentence_array):
    ''' Compute vocabulary

        :param sentence_array: a list of sentences
        :returns: a list of tokens
    '''
    data_vocabulary = {}
    total = 0
    
    for sentence in sentence_array:
        for token in sentence.strip().split():
            if token not in data_vocabulary:
                data_vocabulary[token] = 1 #/len(line.strip().split())
            else:
                data_vocabulary[token] += 1 #/len(line.strip().split())
            total += 1
            
    return total, data_vocabulary

def compute_ld_metric(metric_func, sentences, sample_idxs, iters):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    # 5. let's get the measurements for each sample
    scores = Parallel(n_jobs=-1)(delayed(eval(metric_func))([sentences[j] for j in sample_idxs[i]]) for i in range(iters))
             
    return scores
    
    
def compute_shan_metric(metric_func, sentences, nlpD, sample_idxs = None, iters = 1):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    sample_indexes = []
    if sample_idxs is None:
        sample_indexes.append(range(len(sentences)))
    else:
        sample_indexes = sample_idxs
        
    tmp_lemmas = get_lemmas(sentences, nlpD)
    return eval(metric_func)(tmp_lemmas)
    
    # 5. let's get the measurements for each sample
    #scores = Parallel(n_jobs=-1)(delayed(eval(metric_func))([sentences[j] \
    #    for j in sample_indexes[i]]) for i in range(iters))
             
    #return scores
