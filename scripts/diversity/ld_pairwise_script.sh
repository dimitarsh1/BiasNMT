#!/bin/bash
A=$1 #this is the language pair

mkdir results_ld -p

ld_pairwise.py -f \
	train-${A}-ORGNL.tok \
	train-${A}-RBMT.unk.tok \
	train-${A}-RBMT.unk.tok.onl \
	train-${A}-SMT.out.tok \
	train-${A}-SMT-UNKN.out.tok \
	train-${A}-LSTM-BPE.out.tok \
	train-${A}-LSTM-BPE.out.tok.nobpe \
	train-${A}-TRANS-BPE.out.tok \
	train-${A}-TRANS-BPE.out.tok.nobpe \
	-i 100 -s 1000 > results/ld_$A.out;

ld_pairwise.py -f \
    train-${A}-SMT.back.out.tok \
    train-${A}-SMT-UNKN.back.out.tok \
    train-${A}-LSTM-BPE.back.out.tok \
    train-${A}-LSTM-BPE.back.out.tok.nobpe \
    train-${A}-TRANS-BPE.back.out.tok \
    train-${A}-TRANS-BPE.back.out.tok.nobpe \
    -i 100 -s 1000 > results/ld_${A}_back.out;

