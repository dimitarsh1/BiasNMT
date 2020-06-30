import json
import requests
import xml.etree.ElementTree as ET

class EnglishSpanishTranslator():
    def __init__(self):
        self.englishSpanishDictionary = {}

        root = ET.parse('en-es.xml').getroot()
        for letter in root:
            for word in letter:
                engWord = word.find('c').text
                spaWord = word.find('d').text
                description = word.find('t').text
                partOfSpeech = description[description.find('{')+1:description.find('}')]
                referenceEngWord = None
                if spaWord is None:
                    if 'SEE:' in description:
                        referenceEngWord = description.split('(')[-1][:description.split('(')[-1].find(')')]
                else:
                    spaWord = spaWord.replace('{m}', '').replace('{f}', '')

                wordTransEntry = {
                    'spanish': spaWord,
                    'engRef': referenceEngWord
                }

                if engWord not in self.englishSpanishDictionary:
                    self.englishSpanishDictionary[engWord] = {
                        partOfSpeech: [wordTransEntry]
                    }
                else:
                    if partOfSpeech not in self.englishSpanishDictionary[engWord]:
                        self.englishSpanishDictionary[engWord][partOfSpeech] = [wordTransEntry]
                    else:
                        self.englishSpanishDictionary[engWord][partOfSpeech].append(wordTransEntry)

    def getSpanishTranslations(self, word):
        req_string = 'https://www.dictionaryapi.com/api/v3/references/spanish/json/' + word + '?key=a7654639-8dc3-4723-b0a6-7b838d878d25'
        response = requests.get(req_string)
        response_json = json.loads(response.content)[0]
        
        words = []
        if 'def' in response_json:
            if type(response_json) is str:
                return response_json

            for sense in response_json['def'][0]['sseq']:
                if 'dt' in sense[0][1]:
                    dt = sense[0][1]['dt'][0][1]

                    if type(dt) == list:
                        continue
                    while dt.find('a_link') != -1:
                        idx = dt.find('a_link') + 7
                        dt = dt[idx:]
                        pos_ends = []
                        pos_ends.append(dt.find('}'))
                        pos_ends.append(dt.find(' ('))
                        pos_ends.append(dt.find(','))
                        real_pos_ends = []
                        for end in pos_ends:
                            if end != -1:
                                real_pos_ends.append(end)
                        end = min(real_pos_ends)
                        word = dt[:end]
                        dt = dt[end:]
                        words.append(word)
                
        return list(set(words))

    def getReferenceWordTrans(self, word, pos):
        transWords = []
        if word not in self.englishSpanishDictionary:
            return transWords
        elif pos not in self.englishSpanishDictionary[word]:
            return transWords
        entries = self.englishSpanishDictionary[word][pos]
        for entry in entries:
            spanishEntry = entry['spanish']
            if spanishEntry is not None:
                if ',' in spanishEntry:
                    transWords.extend(spanishEntry.split(','))
                else:
                    transWords.append(spanishEntry)
            else:
                if entry['engRef'] != word:
                    transWords.extend(self.getReferenceWordTrans(entry['engRef'], pos))
        return transWords

    def getSpanishTranslationsFromDict(self, word, pos):
        trans = []
        pos = pos.lower()
        if pos == 'noun':
            pos = 'n'
        elif pos == 'verb':
            pos = 'v'
        if word not in self.englishSpanishDictionary:
            return trans
        elif pos not in self.englishSpanishDictionary[word]:
            return trans
        entries = self.englishSpanishDictionary[word][pos]
        for entry in entries:
            spanishEntry = entry['spanish']
            if spanishEntry is None and entry['engRef'] is not None:
                trans.extend(self.getReferenceWordTrans(entry['engRef'].split('/')[0], pos))
            elif spanishEntry is not None:
                if ',' in spanishEntry:
                    trans.extend([word.strip() for word in spanishEntry.split(',')])
                else:
                    trans.append(spanishEntry)
        return list(set(trans))
