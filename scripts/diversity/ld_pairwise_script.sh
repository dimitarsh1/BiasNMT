#!/bin/bash
A=$1 #this is the language pair

mkdir results_ld -p


ld_pairwise.py -f \
    train-${A}-ORGNL.tok \
	train-${A}-RBMT.unk.tok \
	train-${A}-RBMT.unk.tok.onl \
	train-${A}-RBMT.unk.tok.noerr \
	train-${A}-SMT.out.tok \
	train-${A}-SMT-UNKN.out.tok \
	train-${A}-LSTM-BPE.out.tok \
	train-${A}-LSTM-BPE.out.tok.nobpe \
	train-${A}-TRANS-BPE.out.tok \
	train-${A}-TRANS-BPE.out.tok.nobpe > results_ld/ld_$A.out;

ld_pairwise.py -f \
    train-${A}-SMT.back.out.tok \
    train-${A}-SMT-NODUP-UNKN.back.out.tok \
    train-${A}-LSTM-BPE.back.out.tok \
    train-${A}-LSTM-BPE.back.out.tok.nobpe \
    train-${A}-TRANS-BPE.back.out.tok \
    train-${A}-TRANS-BPE.back.out.tok.nobpe > results_ld/ld_${A}_back.out;

