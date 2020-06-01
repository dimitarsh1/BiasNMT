#!/bin/bash
A=$1
echo $A

mkdir results_lfp -p

lexical_profile.py -f train-${A}.src.tok > results_lfp/lfp_train-${A}_ORGNL.out &
lexical_profile.py -f train-${A}-RBMT.unk.tok.onl > results_lfp/lfp_train-${A}_RBMT_UNKN.out &
lexical_profile.py -f train-${A}-SMT-UNKN.out.tok > results_lfp/lfp_train-${A}_SMT_UNKN.out &
lexical_profile.py -f train-${A}-LSTM-BPE.out.tok.nobpe > results_lfp/lfp_train-${A}_LSTM_NOBPE.out &
lexical_profile.py -f train-${A}-TRANS-BPE.out.tok.nobpe > results_lfp/lfp_train-${A}_TRANS_NOBPE.out &

# backtranslated systems:
lexical_profile.py -f train-${A}-SMT-NODUP-UNKN.back.out.tok > results_lfp/lfp_back_train-${A}_SMT_UNKN.out &
lexical_profile.py -f train-${A}-LSTM-BPE.back.out.tok.nobpe > results_lfp/lfp_back_train-${A}_LSTM_NOBPE.out &
lexical_profile.py -f train-${A}-TRANS-BPE.back.out.tok.nobpe > results_lfp/lfp_back_train-${A}_TRANS_NOBPE.out &
wait

