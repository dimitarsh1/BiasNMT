#!/bin/bash
# Training an NMT system: SMT, LSTM or Transformer
# train.sh SYSTEM DATADIR SOURCE TARGET
#          SYSTEM: one of SMT, LSTM or Transformer
#          DATADIR: the directory containing the data. Data is in format train/test/dev.src/.trg
#          SOURCE: the source language (e.g., en)
#          TARGET: th target language (e.g., fr)
#          ENGINEDIR: the working directory, where files and models will be stored

SYSTEM=$1
ENGINEDIR=$2
SRC=$3
TRG=$4
INPUT=$5

MODELDIR=${ENGINEDIR}/model

echo 'Engine dir: ' $ENGINEDIR

export ENGINEDIR=$ENGINEDIR
export MODELDIR=$MODELDIR
export SRCLANG=$SRC
export TRGLANG=$TRG
echo 'Variables exported'

MTPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )


if [ $SYSTEM == 'SMT' ]
then
    $MTPATH/../external/SMT/3_translate_moses.sh --enginedir $ENGINEDIR --source $SRC --target $TRG --input $ENGINEDIR/data/$INPUT
    echo 'Translating with Moses finished'
    exit 0
fi

if [ $SYSTEM == 'LSTM' ]
then
    NMTPATH=$MTPATH/../external/NMT
    # translate with the already trained system
    $NMTPATH/MTTools/5_translate.sh $ENGINEDIR $INPUT
    echo 'Translating with LSTM finished'

    exit 0
fi

if [ $SYSTEM == 'TRANS' ]
then
    NMTPATH=$MTPATH/../external/NMT
    # translate with the already trained system
    $NMTPATH/5_translate.sh $ENGINEDIR $INPUT
    echo 'Translating with Transformer finished'

    exit 0
fi

