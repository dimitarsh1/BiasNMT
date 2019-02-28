#!/bin/bash
# Training an NMT system: SMT, LSTM or Transformer
# train.sh SYSTEM DATADIR SOURCE TARGET
#          SYSTEM: one of SMT, LSTM or Transformer
#          DATADIR: the directory containing the data. Data is in format train/test/dev.src/.trg
#          SOURCE: the source language (e.g., en)
#          TARGET: th target language (e.g., fr)
#          ENGINEDIR: the working directory, where files and models will be stored

SYSTEM=$1
DATADIR=$2
SRC=$3
TRG=$4
ENGINEDIR=$5

if [ -z $ENGINEDIR ]
then
    ENGINEDIR=${SRC}'-'${TRG}'-'${SYSTEM}
fi
echo $ENGINEDIR

mkdir -p $ENGINEDIR
mkdir -p $ENGINEDIR/data
mkdir -p $ENGINEDIR/model

echo 'Engine dir created: ' $ENGINEDIR

export ENGINEDIR=$ENGINEDIR
export SRCLANG=$SRC
export TRGLANG=$TRG
echo 'Variables exported'

for ext in src trg
do
    for f in test dev
    do
        cp $DATADIR/${f}.tok.${ext} $ENGINEDIR/data/
    done
    cp $DATADIR/train.clean.${ext} $ENGINEDIR/data/
done

echo 'Files transfered to working dir'

MTPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
echo 'MTPath is: ' $MTPATH


if [ $SYSTEM == 'SMT' ]
then
    $MTPATH/../external/SMT/2_train_moses.sh --datadir $ENGINEDIR --source $SRC --target $TRG
    echo 'Training Moses finished'
    exit 0
fi

if [ $SYSTEM == 'LSTM' ]
then
    NMTPATH=$MTPATH/../external/NMT
    # create dictionary
    $NMTPATH/3_word_dictionary.sh
    # train the system
    $NMTPATH/MTTools/4_train_lstm_words.sh $ENGINEDIR
    echo 'Training LSTM finished'

    exit 0
fi

if [ $SYSTEM == 'TRANS' ]
then
    NMTPATH=$MTPATH/../external/NMT
    # create dictionary
    $NMTPATH/3_word_dictionary.sh
    # train the system
    $NMTPATH/4_train_trans_words.sh $ENGINEDIR
    echo 'Training Transformer finished'

    exit 0
fi

