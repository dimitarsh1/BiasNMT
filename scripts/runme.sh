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

    read -p "Press any key to continue... "
    # Forward systems
    for S in $SYSTEMS
    do
        # 2. Train forward system
        #    $SCRIPTPATH/train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        MODELDIR=${SCRIPTPATH}/../data/${SRC}'-'${TRG}'-'${S}
        $SCRIPTPATH/train.sh $S ${SCRIPTPATH}/../data/$LANGPAIR $SRC $TRG $MODELDIR
        echo 'Training first moses system - done'
        read -p "Press any key to continue... "

        # 3. Translate with forward system (thus getting the distribution after one MT engine)
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${MODELDIR} $SRC $TRG train
        $SCRIPTPATH/translate.sh $S ${MODELDIR} $SRC $TRG test

        # 4. Score translated test
        #    score.sh FILETOSCORE
        python3 $SCRIPTPATH/score.py -i ${MODELDIR}/train.clean.${TRG}.out -o ${RESDIR}/forward.train.score
        python3 $SCRIPTPATH/score.py -i ${MODELDIR}/test.clean.${TRG}.out -o ${RESDIR}/forward.test.score

        # 5. Train reversed system to use for backtranslating data
        #    $SCRIPTPATH/train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        REVMODELDIR=${SCRIPTPATH}/../data/${TRG}'-'${SRC}'-'${S}
        $SCRIPTPATH/train.sh $S ${SCRIPTPATH}/../data/$LANGPAIR $TRG $SRC ${REVMODELDIR}

        # 6. Translate with the reverse system
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${REVMODELDIR} $TRG $SRC train

        # 7. Copy backtranslated data and original data to new folder
        BACKMODELDIR=${SCRIPTPATH}/../data/${SRC}'-'${TRG}'-'${S}'-BACK'
        mkdir ${BACKMODELDIR} -p
        cp ${REVMODELDIR}/train.clean.${TRG}.out ${BACKMODELDIR}/train.clean.${TRG}
        cp ${REVMODELDIR}/test.clean.${TRG}.out ${BACKMODELDIR}/test.clean.${TRG}
        cp ${REVMODELDIR}/dev.clean.${TRG}.out ${BACKMODELDIR}/dev.clean.${TRG}

        # 8. Train system with backtranslated data
        $SCRIPTPATH/train.sh $S ${BACKMODELDIR} $SRC $TRG

        # 9. Translate text with the backtranslated models
        #    $SCRIPTPATH/translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        $SCRIPTPATH/translate.sh $S ${BACKMODELDIR} train
        $SCRIPTPATH/translate.sh $S ${BACKMODELDIR} test

        # 10. Score the translated text with the backtranslated model
        #     score.sh FILETOSCORE
        python3 $SCRIPTPATH/score.py -i ${BACKMODELDIR}/train.clean.${TRG}.out -o ${RESDIR}/backward.train.score
        python3 $SCRIPTPATH/score.py -i ${BACKMODELDIR}/test.clean.${TRG}.out -o ${RESDIR}/backward.test.score
    done
done
