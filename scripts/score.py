# -*- coding: utf-8 -*-

''' reads a text file and exports unique tokens separated by space and their frequencies.
'''
import argparse
import codecs
import os
import sys

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Extracts words to a dictionary with their frequencies.')
    parser.add_argument('-i', '--input', required=True, help='the data file to read.')
    parser.add_argument('-o', '--output', required=True, help='the data file to read.')

    args = parser.parse_args()

    # initialize a vocabulary as a set (only one occurrence of an element)
    data_vocabulary = {}
    if os.path.exists(args.input):
        with codecs.open(args.input, 'r', 'utf8') as fh:
            for line in fh:
                # do the union to add more elements to the vocabulary
                for token in line.split():
                    if token not in data_vocabulary:
                        data_vocabulary[token] = 1
                    else:
                        data_vocabulary[token] += 1

        # print the vocabulary to the file.
        with codecs.open(args.outfile, 'w', 'utf8') as ofh:
            ofh.write("\n".join([x + '\t' + str(a[x]) for x in sorted(a, key=a.get, reverse=True)]) + '\n')
    else:
        # if file doesn't exist exit and print a message to stderr`
        print("ERROR: Path not found: ", args.input, file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    main()
