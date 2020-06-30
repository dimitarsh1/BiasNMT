import json
import requests
import xml.etree.ElementTree as ET

class EnglishFrenchTranslator():
    def __init__(self):
        self.dictionary = {}
       
        with open('posDict.json', 'r') as jsonFile:
            partsOfSpeechDict = json.load(jsonFile)

        root = ET.parse('eng-fra.xml').getroot()
        rootTag = root.tag.replace('TEI', '')
        for entry in root[1][0]:
            srcWord = entry[0].find(rootTag + 'orth').text
            for tgtEntry in entry[1:]:
                cit = tgtEntry.find(rootTag + 'cit')
                entryType = cit.attrib['type']
                if entryType == 'trans':
                    tgtWord = cit.find(rootTag + 'quote').text
                    if tgtWord in partsOfSpeechDict:
                        partsOfSpeech = partsOfSpeechDict[tgtWord]
                        for partOfSpeech in partsOfSpeech:
                            print(srcWord + ', ' + tgtWord + ', ' + partOfSpeech)
                            if srcWord not in self.dictionary:
                                self.dictionary[srcWord] = {
                                    partOfSpeech: [tgtWord]
                                }
                            else:
                                if partOfSpeech not in self.dictionary[srcWord]:
                                    self.dictionary[srcWord][partOfSpeech] = [tgtWord]
                                else:
                                    self.dictionary[srcWord][partOfSpeech].append(tgtWord)

    def getTranslations(self, word, pos):
        trans = []

        if word not in self.dictionary:
            return trans
        elif pos not in self.dictionary[word]:
            return trans
        else:
            entries = self.dictionary[word][pos]
            for tgtEntry in entries:
                if ',' in tgtEntry:
                    trans.extend([word.strip() for word in tgtEntry.split(',')])
                else:
                    trans.append(tgtEntry)
            return list(set(trans))


englishFrenchTranslator = EnglishFrenchTranslator()
