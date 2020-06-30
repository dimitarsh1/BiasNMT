import pickle
import operator
import string
import spacy

#RUN the following:
#python -m spacy download en
#python -m spacy download es

en_nlp = spacy.load('en')
es_nlp = spacy.load('es')

def getLemmaList(sentenceList, langSpacy):
  lemmaList = []
  posList = []
  for i, sentence in enumerate(sentenceList):
    doc = langSpacy(sentence)
    lemmas = [token.lemma_ for token in doc]
    partsOfSpeech = [token.pos_ for token in doc]
    lemmaList.append(lemmas)
    posList.append(partsOfSpeech)

    if i % 500 == 0:
      print(i)

  return lemmaList, posList

def getLemmaListAndFrequencyDictSorted(sentenceList, langSpacy, goodPartsOfSpeech = ['ADJ'], badTokens = ['-PRON-', '\n']):
  badTokens.extend([punct for punct in string.punctuation])
  
  lemmaFrequencyDict = {}

  lemmaList = []
  posList = []
  for i, sentence in enumerate(sentenceList):
    doc = langSpacy(sentence)

    lemmas = [token.lemma_ for token in doc]
    partsOfSpeech = [token.pos_ for token in doc]
    lemmaList.append(lemmas)
    posList.append(partsOfSpeech)

    for token in doc:
      lem = token.lemma_
      pos = token.pos_
      if lem in badTokens or pos not in goodPartsOfSpeech:
        continue
      lem = lem.lower()
      if lem not in lemmaFrequencyDict:
        lemmaFrequencyDict[lem] = 0
      lemmaFrequencyDict[lem] += 1

    if i % 500 == 0:
      print(i)

  lemmaFrequencyDictSorted = sorted(lemmaFrequencyDict.items(), key=operator.itemgetter(1), reverse=True)

  for (word, freq) in lemmaFrequencyDictSorted[:10]:
    print(word, freq)

  return lemmaFrequencyDictSorted, lemmaList, posList

with open('data/train-es-en-ORGNL.tok', 'r') as fileToRead:
  trainEnglishSrc = fileToRead.readlines()

with open('data/train-en-es-ORGNL.tok', 'r') as fileToRead:
  trainSpanishTgt = fileToRead.readlines()

with open('data/train-en-es-TRANS-BPE.out.tok.nobpe', 'r') as fileToRead:
  trainSpanishTrans = fileToRead.readlines()

with open('data/train-en-es-TRANS-BPE.back.out.tok.nobpe', 'r') as fileToRead:
  trainSpanishTranBack = fileToRead.readlines()

with open('data/train-en-es-LSTM-BPE.out.tok.nobpe', 'r') as fileToRead:
  trainSpanishLSTM = fileToRead.readlines()

with open('data/train-en-es-LSTM-BPE.back.out.tok.nobpe', 'r') as fileToRead:
  trainSpanishLSTMBack = fileToRead.readlines()

with open('data/train-en-es-SMT-UNKN.out.tok', 'r') as fileToRead:
  trainSpanishSMT = fileToRead.readlines()

with open('data/train-en-es-SMT-NODUP-UNKN.back.out.tok', 'r') as fileToRead:
  trainSpanishSMTBack = fileToRead.readlines()

with open('data/train-en-es-RBMT.unk.tok.noerr', 'r') as fileToRead:
  trainSpanishRBMTNoErr = fileToRead.readlines()

with open('data/train-en-es-RBMT.unk.tok.onl', 'r') as fileToRead:
  trainSpanishRBMTOnl = fileToRead.readlines()


srcLemFreqDictSort, srcLemSents, sourcePOS = getLemmaListAndFrequencyDictSorted(trainEnglishSrc, en_nlp, ['ADJ', 'NOUN', 'VERB'])
with open('sourceLemmaFreqDictSorted.pkl', 'wb') as output:
  pickle.dump(srcLemFreqDictSort, output)
with open('sourceLemmatizedSentences.pkl', 'wb') as output:
  pickle.dump(srcLemSents, output)
with open('sourcePartsOfSpeech.pkl', 'wb') as output:
  pickle.dump(sourcePOS, output)

tgtLemSentsRef, tgtPOSRef = getLemmaList(trainSpanishTgt, es_nlp)
with open('targetLemmatizedSentencesRef.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsRef, output)
with open('targetPartsOfSpeechRef.pkl', 'wb') as output:
  pickle.dump(tgtPOSRef, output)

tgtLemSentsTrans, tgtPOSTrans = getLemmaList(trainSpanishTrans, es_nlp)
with open('targetLemmatizedSentencesTrans.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsTrans, output)
with open('targetPartsOfSpeechTrans.pkl', 'wb') as output:
  pickle.dump(tgtPOSTrans, output)

tgtLemSentsTransBack, tgtPOSTransBack = getLemmaList(trainSpanishTransBack, es_nlp)
with open('targetLemmatizedSentencesTransBack.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsTransBack, output)
with open('targetPartsOfSpeechTransBack.pkl', 'wb') as output:
  pickle.dump(tgtPOSTransBack, output)

tgtLemSentsLSTM, tgtPOSLSTM = getLemmaList(trainSpanishLSTM, es_nlp)
with open('targetLemmatizedSentencesLSTM.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsLSTM, output)
with open('targetPartsOfSpeechLSTM.pkl', 'wb') as output:
  pickle.dump(tgtPOSLSTM, output)

tgtLemSentsLSTMBack, tgtPOSLSTMBack = getLemmaList(trainSpanishLSTMBack, es_nlp)
with open('targetLemmatizedSentencesLSTMBack.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsLSTMBack, output)
with open('targetPartsOfSpeechLSTMBack.pkl', 'wb') as output:
  pickle.dump(tgtPOSLSTMBack, output)

tgtLemSentsSMT, tgtPOSSMT = getLemmaList(trainSpanishSMT, es_nlp)
with open('targetLemmatizedSentencesSMT.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsSMT, output)
with open('targetPartsOfSpeechSMT.pkl', 'wb') as output:
  pickle.dump(tgtPOSSMT, output)

tgtLemSentsSMTBack, tgtPOSSMTBack = getLemmaList(trainSpanishSMTBack, es_nlp)
with open('targetLemmatizedSentencesSMTBack.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsSMTBack, output)
with open('targetPartsOfSpeechSMTBack.pkl', 'wb') as output:
  pickle.dump(tgtPOSSMTBack, output)

tgtLemSentsRBMTNoErr, tgtPOSRBMTNoErr = getLemmaList(trainSpanishRBMTNoErr, es_nlp)
with open('targetLemmatizedSentencesRBMTNoErr.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsRBMTNoErr, output)
with open('targetPartsOfSpeechRBMTNoErr.pkl', 'wb') as output:
  pickle.dump(tgtPOSRBMTNoErr, output)

tgtLemSentsRBMTOnl, tgtPOSRBMTOnl = getLemmaList(trainSpanishRBMTOnl, es_nlp)
with open('targetLemmatizedSentencesRBMTOnl.pkl', 'wb') as output:
  pickle.dump(tgtLemSentsRBMTOnl, output)
with open('targetPartsOfSpeechRBMTOnl.pkl', 'wb') as output:
  pickle.dump(tgtPOSRBMTOnl, output)
