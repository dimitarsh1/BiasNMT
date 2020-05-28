#!/bin/bash
A=$1 #this is the language pair
LANG=`echo ${A} | cut -d '-' -f 2`
echo $LANG

mkdir results -p

shannon_pairwise.py -f train-${A}-SMT-NODUP-UNKN.back.out.tok -l $LANG > results/shannon_back_train-${A}_SMT_UNKN.out
shannon_pairwise.py -f train-${A}-LSTM-BPE.back.out.tok.nobpe -l $LANG > results/shannon_back_train-${A}_LSTM_NOBPE.out
shannon_pairwise.py -f train-${A}-TRANS-BPE.back.out.tok.nobpe -l $LANG > results/shannon_back_train-${A}_TRANS_NOBPE.out
