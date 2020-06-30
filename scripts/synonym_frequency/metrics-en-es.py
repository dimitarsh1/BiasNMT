from scipy import spatial
import numpy as np
import pickle

def getPrimaryPercent(transFreqDict, translatedCountsOnly = False):
  primaryCount = 0
  totalCount = 0
  for word, info in transFreqDict.items():
    if len(info['translations']) < 1:# or info['count'] > 1000:
      continue
    
    primaryCount += info['translations'][0][1]
    curTotalCount = 0
    if translatedCountsOnly:
      for translationFrequency in info['translations']:
        curTotalCount += translationFrequency[1]
      if curTotalCount > 0 and info['translations'][0][1] / curTotalCount < 0.40:
        #print(word)
        #print(info)
      totalCount += curTotalCount
    else:
      totalCount += info['count']
  #print(primaryCount)
  #print(totalCount)
  return primaryCount / totalCount

def getDifferenceBetweenTranslations(transFreqDictA, transFreqDictB):
  counted = 0
  totalDistance = 0
  for word, infoA in transFreqDictA.items():
    if len(infoA['translations']) < 2:# or info['count'] > 1000:
      continue
    
    infoB = transFreqDictB[word]
    counted += 1
    transFreqsA = []
    for translationFrequency in infoA['translations']:
      transFreqsA.append(translationFrequency[1])
    transFreqsB = []
    for translationFrequency in infoB['translations']:
      transFreqsB.append(translationFrequency[1])
    transCountA = sum(transFreqsA)
    transCountB = sum(transFreqsB)
    if transCountA > 0 and transCountB > 0:
      transFreqsA = np.array(transFreqsA) / transCountA
      transFreqsB = np.array(transFreqsB) / transCountB
      distance = spatial.distance.cosine(transFreqsA, transFreqsB)
      totalDistance += distance
    else:
      totalDistance += 1.0
  #print(totalDistance)
  return totalDistance / counted

def getSynonymTTR(transFreqDict):
  counted = 0
  numTypes = 0
  numTokens = 0
  totalUnweightedTTR = 0.0
  for word, info in transFreqDict.items():
    if len(info['translations']) < 1:# or info['count'] > 1000:
      continue
    
    curNumTokens = 0
    curNumTypes = 0
    for translationFrequency in info['translations']:
      curNumTokens += translationFrequency[1]
      if translationFrequency[1] > 0:
        curNumTypes += 1

    if curNumTokens > 0:
      counted += 1
      totalUnweightedTTR += (curNumTypes / curNumTokens) 
      numTokens += curNumTokens
      numTypes += curNumTypes

  #print(numTypes)
  #print(numTokens)
  return numTypes / numTokens, totalUnweightedTTR / counted

def getDifferenceFromUniform(transFreqDict):
  counted = 0
  totalDistance = 0
  for word, info in transFreqDict.items():
    if len(info['translations']) < 2:# or info['count'] > 1000:
      continue
    
    counted += 1
    transFreqs = []
    for translationFrequency in info['translations']:
      transFreqs.append(translationFrequency[1])
    transCount = sum(transFreqs)
    if transCount > 0:
      transFreqs = np.array(transFreqs) / transCount
      #uniformFreqs = np.ones(len(info['translations'])) / len(info['translations'])
      uniformFreqs = np.ones(len(info['translations']))
      distance = spatial.distance.cosine(uniformFreqs, transFreqs)
      totalDistance += distance
    else:
      totalDistance += 1.0
  #print(totalDistance)
  return totalDistance / counted

with open('transFreqDictRef.pkl', 'rb') as infile:
  transFreqDictRef = pickle.load(infile)
with open('transFreqDictTrans.pkl', 'rb') as infile:
  transFreqDictTrans = pickle.load(infile)
with open('transFreqDictLSTM.pkl', 'rb') as infile:
  transFreqDictLSTM = pickle.load(infile)
with open('transFreqDictSMT.pkl', 'rb') as infile:
  transFreqDictSMT = pickle.load(infile)
with open('transFreqDictTransBack.pkl', 'wb') as infile:
  transFreqDictTransBack = pickle.load(infile)
with open('transFreqDictLSTMBack.pkl', 'wb') as infile:
  transFreqDictLSTMBack = pickle.load(infile)
with open('transFreqDictSMTBack.pkl', 'wb') as infile:
  transFreqDictSMTBack = pickle.load(infile)
with open('transFreqDictRBMTNoErr.pkl', 'wb') as infile:
  transFreqDictRBMTNoErr = pickle.load(infile)
with open('transFreqDictRBMTOnl.pkl', 'wb') as infile:
  transFreqDictRBMTOnl = pickle.load(infile)

"""
print('Compute Relative Differences (differences between translations)')

differenceRefTrans = getDifferenceBetweenTranslations(transFreqDictRef, transFreqDictTrans)
differenceRefLSTM = getDifferenceBetweenTranslations(transFreqDictRef, transFreqDictLSTM)
differenceRefSMT = getDifferenceBetweenTranslations(transFreqDictRef, transFreqDictSMT)

print(differenceRefTrans)
print(differenceRefLSTM)
print(differenceRefSMT)
"""

print('Compute type token ratio for synoynms only, weighted and unweighted')

synonymTTRRef, unweightedSynonymTTRRef = getSynonymTTR(transFreqDictRef)
synonymTTRTrans, unweightedSynonymTTRTrans = getSynonymTTR(transFreqDictTrans)
synonymTTRLSTM, unweightedSynonymTTRLSTM = getSynonymTTR(transFreqDictLSTM)
synonymTTRSMT, unweightedSynonymTTRSMT = getSynonymTTR(transFreqDictSMT)
synonymTTRTransBack, unweightedSynonymTTRTransBack = getSynonymTTR(transFreqDictTransBack)
synonymTTRLSTMBack, unweightedSynonymTTRLSTMBack = getSynonymTTR(transFreqDictLSTMBack)
synonymTTRSMTBack, unweightedSynonymTTRSMTBack = getSynonymTTR(transFreqDictSMTBack)
synonymTTRRBMTNoErr, unweightedSynonymTTRRBMTNoErr = getSynonymTTR(transFreqDictRBMTNoErr)
synonymTTRRBMTOnl, unweightedSynonymTTRRBMTOnl = getSynonymTTR(transFreqDictRBMTOnl)

print(synonymTTRRef, unweightedSynonymTTRRef)
print(synonymTTRTrans, unweightedSynonymTTRTrans)
print(synonymTTRLSTM, unweightedSynonymTTRLSTM)
print(synonymTTRSMT, unweightedSynonymTTRSMT)
print(synonymTTRTransBack, unweightedSynonymTTRTransBack)
print(synonymTTRLSTMBack, unweightedSynonymTTRLSTMBack)
print(synonymTTRSMTBack, unweightedSynonymTTRSMTBack)

print('Cosine difference between uniform and actual synoynm distribution')

uniformDistanceRef = getDifferenceFromUniform(transFreqDictRef)
uniformDistanceTrans = getDifferenceFromUniform(transFreqDictTrans)
uniformDistanceLSTM = getDifferenceFromUniform(transFreqDictLSTM)
uniformDistanceSMT = getDifferenceFromUniform(transFreqDictSMT)
uniformDistanceTransBack = getDifferenceFromUniform(transFreqDictTransBack)
uniformDistanceLSTMBack = getDifferenceFromUniform(transFreqDictLSTMBack)
uniformDistanceSMTBack = getDifferenceFromUniform(transFreqDictSMTBack)
uniformDistanceRBMTNoErr = getDifferenceFromUniform(transFreqDictRBMTNoErr)
uniformDistanceRBMTOnl = getDifferenceFromUniform(transFreqDictRBMTOnl)

print(uniformDistanceRef)
print(uniformDistanceTrans)
print(uniformDistanceLSTM)
print(uniformDistanceSMT)
print(uniformDistanceTransBack)
print(uniformDistanceLSTMBack)
print(uniformDistanceSMTBack)
print(uniformDistanceRBMTNoErr)
print(uniformDistanceRBMTOnl)

"""
print('Percentage a token was translated to primary word/phrase, raw')
primaryPercentRef = getPrimaryPercent(transFreqDictRef)
primaryPercentTrans = getPrimaryPercent(transFreqDictTrans)
primaryPercentLSTM = getPrimaryPercent(transFreqDictLSTM)
primaryPercentSMT = getPrimaryPercent(transFreqDictSMT)

print(primaryPercentRef)
print(primaryPercentTrans)
print(primaryPercentLSTM)
print(primaryPercentSMT)
"""

print('Percentage a token was translated to primary word/phrase, only counting if translation identified')
primaryPercentRef = getPrimaryPercent(transFreqDictRef, True)
primaryPercentTrans = getPrimaryPercent(transFreqDictTrans, True)
primaryPercentLSTM = getPrimaryPercent(transFreqDictLSTM, True)
primaryPercentSMT = getPrimaryPercent(transFreqDictSMT, True)
primaryPercentTransBack = getPrimaryPercent(transFreqDictTransBack, True)
primaryPercentLSTMBack = getPrimaryPercent(transFreqDictLSTMBack, True)
primaryPercentSMTBack = getPrimaryPercent(transFreqDictSMTBack, True)
primaryPercentRBMTNoErr = getPrimaryPercent(transFreqDictRBMTNoErr, True)
primaryPercentRBMTOnl = getPrimaryPercent(transFreqDictRBMTOnl, True)

print(primaryPercentRef)
print(primaryPercentTrans)
print(primaryPercentLSTM)
print(primaryPercentSMT)
print(primaryPercentTransBack)
print(primaryPercentLSTMBack)
print(primaryPercentSMTBack)
print(primaryPercentRBMTNoErr)
print(primaryPercentRBMTOnl)