#!/bin/bash
A=$1 #this is the language pair
LANG=`echo ${A} | cut -d '-' -f 2`
echo $LANG

mkdir results -p

shannon_pairwise.py -f train-${A}.src.tok -l $LANG > results/shannon_train-${A}_ORGNL.out &
shannon_pairwise.py -f train-${A}-RBMT.unk.tok.onl -l $LANG > results/shannon_train-${A}_RBMT_UNKN.out &
shannon_pairwise.py -f train-${A}-SMT-UNKN.out.tok -l $LANG > results/shannon_train-${A}_SMT_UNKN.out &
shannon_pairwise.py -f train-${A}-LSTM-BPE.out.tok.nobpe -l $LANG > results/shannon_train-${A}_LSTM_NOBPE.out &
shannon_pairwise.py -f train-${A}-TRANS-BPE.out.tok.nobpe -l $LANG > results/shannon_train-${A}_TRANS_NOBPE.out &

wait
