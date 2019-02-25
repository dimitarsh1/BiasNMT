#!/bin/bash

DATADIR=$1
SRCLANG=$2
TRGLANG=$3

MTPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
echo "System path ${MTPATH}"
TOKENIZER=$MTPATH/../external/SMT/mosesdecoder/scripts/tokenizer
TRAINING=$MTPATH/../external/SMT/mosesdecoder/scripts/training


$TOKENIZER/tokenizer.perl -l $SRCLANG < $DATADIR/europarl.${SRCLANG}.shuf.dev > $DATADIR/dev.tok.src
$TOKENIZER/tokenizer.perl -l $TRGLANG < $DATADIR/europarl.${TRGLANG}.shuf.dev > $DATADIR/dev.tok.trg

$TOKENIZER/tokenizer.perl -l $SRCLANG < $DATADIR/europarl.${SRCLANG}.shuf.test > $DATADIR/test.tok.src
$TOKENIZER/tokenizer.perl -l $TRGLANG < $DATADIR/europarl.${TRGLANG}.shuf.test > $DATADIR/test.tok.trg

$TOKENIZER/tokenizer.perl -l $SRCLANG < $DATADIR/europarl.${SRCLANG}.shuf.train > $DATADIR/train.tok.src
$TOKENIZER/tokenizer.perl -l $TRGLANG < $DATADIR/europarl.${TRGLANG}.shuf.train > $DATADIR/train.tok.trg

$TRAINING/clean-corpus-n.perl $DATADIR/train.tok src trg $DATADIR/train.clean 1 80
