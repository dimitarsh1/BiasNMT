#!/bin/bash
A=$1 #the language pair

mkdir results -p

ld_pairwise.py -f \
        train-${A}-SMT.back.out.tok \
        train-${A}-SMT-UNKN.back.out.tok \
        train-${A}-LSTM-BPE.back.out.tok \
        train-${A}-LSTM-BPE.back.out.tok.nobpe \
        train-${A}-TRANS-BPE.back.out.tok \
        train-${A}-TRANS-BPE.back.out.tok.nobpe \
	-i 100 -s 1000 > results/ld_${A}_back.out;
