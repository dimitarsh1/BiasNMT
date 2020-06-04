#!/bin/bash
A=$1 #this is the language pair

mkdir results_ld -p

#	train-${A}-ORGNL.tok \

ld_pairwise.py -f \
	train-${A}-RBMT.unk.tok \
	train-${A}-RBMT.unk.tok.onl \
	train-${A}-RBMT.unk.tok.noerr > results_ld/ld_${A}_RMBT.out #\
#	train-${A}-SMT.out.tok \
#	train-${A}-SMT-UNKN.out.tok \
#	train-${A}-LSTM-BPE.out.tok \
#	train-${A}-LSTM-BPE.out.tok.nobpe \
#	train-${A}-TRANS-BPE.out.tok \
#	train-${A}-TRANS-BPE.out.tok.nobpe > results/ld_$A.out;

#ld_pairwise.py -f \
#    train-${A}-SMT.back.out.tok \
#    train-${A}-SMT-UNKN.back.out.tok \
#    train-${A}-LSTM-BPE.back.out.tok \
#    train-${A}-LSTM-BPE.back.out.tok.nobpe \
#    train-${A}-TRANS-BPE.back.out.tok \
#    train-${A}-TRANS-BPE.back.out.tok.nobpe > results/ld_${A}_back.out;

