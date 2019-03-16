#!/bin/bash

DIRS='EN-FR EN-ES'
SYSTEMS='SMT' # LSTM TRANS'

RESDIR='SCORES'
mkdir ${RESDIR} -p

SCRIPTPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
echo "System path ${SCRIPTPATH}"

# GET Source and Target languages from dirs:
for LANGPAIR in $DIRS
do
    echo $LANGPAIR
    SRC=$( echo ${LANGPAIR,,} | sed 's/-/ /g' | cut -d ' ' -f 1 )
    TRG=$( echo ${LANGPAIR,,} | sed 's/-/ /g' | cut -d ' ' -f 2 )
    echo 'Src: ' $SRC
    echo 'Trg: ' $TRG

    # 1. Score ORIGINAL DATA
    #    score.sh FILETOSCORE
    if [ ! -d ${RESDIR}/${SRC}'-'${TRG}'-original.train.score' ]
    then
        echo 'Scoring original data'
        python3 $SCRIPTPATH/score.py -i ${SCRIPTPATH}/../data/${LANGPAIR}/train.clean.trg -o ${RESDIR}/original.train.score
        python3 $SCRIPTPATH/score.py -i ${SCRIPTPATH}/../data/${LANGPAIR}/test.tok.trg -o ${RESDIR}/original.test.score
    fi

    # Forward systems
    for S in $SYSTEMS
    do
        # 2. Train forward system
        #    $SCRIPTPATH/train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        MODELDIR=${SCRIPTPATH}/../data/${SRC}'-'${TRG}'-'${S}
        #$SCRIPTPATH/train.sh $S ${SCRIPTPATH}/../data/$LANGPAIR $SRC $TRG $MODELDIR
        echo 'Training first moses system - done'

        # 3. Translate with forward system (thus getting the distribution after one MT engine)
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${MODELDIR} $SRC $TRG train.clean.src
        $SCRIPTPATH/translate.sh $S ${MODELDIR} $SRC $TRG test.tok.src

        # 4. Score translated test
        #    score.sh FILETOSCORE
        python3 $SCRIPTPATH/score.py -i ${MODELDIR}/data/train.clean.src.out -o ${RESDIR}/forward.train.score
        python3 $SCRIPTPATH/score.py -i ${MODELDIR}/data/test.tok.src.out -o ${RESDIR}/forward.test.score

        # 5. Train reversed system to use for backtranslating data
        #    $SCRIPTPATH/train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        REVMODELDIR=${SCRIPTPATH}/../data/${TRG}'-'${SRC}'-'${S}
        $SCRIPTPATH/train.sh $S ${SCRIPTPATH}/../data/$LANGPAIR $TRG $SRC ${REVMODELDIR} trg src

        # 6. Translate with the reverse system
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${REVMODELDIR} $TRG $SRC train.clean.src

        # 7. Copy backtranslated data and original data to new folder
        TMPDIR=${SCRIPTPATH}/../data/${SRC}'-'${TRG}'-'${S}'-TMP'
        mkdir ${TMPDIR} -p
        cp ${REVMODELDIR}/train.clean.src.out ${TMPDIR}/train.clean.src
        cp ${REVMODELDIR}/test.tok.trg ${TMPDIR}/test.tok.src
        cp ${REVMODELDIR}/dev.tok.trg ${TMPDIR}/dev.tok.src
        cp ${REVMODELDIR}/test.tok.src ${TMPDIR}/test.tok.trg
        cp ${REVMODELDIR}/dev.tok.src ${TMPDIR}/dev.tok.trg
        cp ${REVMODELDIR}/train.clean.src ${TMPDIR}/train.clean.trg

        # 8. Train system with backtranslated data
        BACKMODELDIR=${SCRIPTPATH}/../data/${SRC}'-'${TRG}'-'${S}'-BACK'
        $SCRIPTPATH/train.sh $S ${TMPDIR} $SRC $TRG ${BACKMODELDIR}

        cp $LANGIR/train.clean.src ${BACKMODELDIR}/data/train.clean.src.orgnl
        cp $LANGIR/test.tok.src ${BACKMODELDIR}/data/test.tok.src.orgnl

        # 9. Translate text with the backtranslated models
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${BACKMODELDIR} ${SRC} ${TRG} train.clean.src.orgnl
        $SCRIPTPATH/translate.sh $S ${BACKMODELDIR} ${SRC} ${TRG} test.tok.src.orgnl

        # 10. Score the translated text with the backtranslated model
        #     score.sh FILETOSCORE
        python3 $SCRIPTPATH/score.py -i ${BACKMODELDIR}/train.clean.src.orgnl.out -o ${RESDIR}/backward.train.score
        python3 $SCRIPTPATH/score.py -i ${BACKMODELDIR}/test.tok.src.orgnl.out -o ${RESDIR}/backward.test.score
    done
done
