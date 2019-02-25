#!/bin/bash

DIRS='EN-FR EN-ES'
SYSTEMS='SMT'# LSTM TRANS'

RESDIR='SCORES'
mkdir ${RESDIR} -p

# GET Source and Target languages from dirs:
for LANGPAIR in $DIRS
do
    echo $LANGPAIR
    SRC=$( echo ${i,,} | sed 's/-/ /g' | cut -d ' ' -f 1 )
    TRG=$( echo ${i,,} | sed 's/-/ /g' | cut -d ' ' -f 2 )
    echo 'Src: ' $SRC
    echo 'Trg: ' $TRG

    # Forward systems
    for S in SYSTEMS
    do
        # 1. Score ORIGINAL DATA
        #    score.sh FILETOSCORE
        python3 score.py -i ${LANGDIR}/train.clean.${TRG} -o ${RESDIR}/original.train.score
        python3 score.py -i ${LANGDIR}/test.clean.${TRG} -o ${RESDIR}/original.test.score

        # 2. Train forward system
        #    train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        MODELDIR=${SRC}'-'${TRG}'-'${S}
        train.sh $S $LANGPAIR $SRC $TRG $MODELDIR

        # 3. Translate with forward system (thus getting the distribution after one MT engine)
        #    translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        translate.sh $S ${MODELDIR} $SRC $TRG train
        translate.sh $S ${MODELDIR} $SRC $TRG test

        # 4. Score translated test
        #    score.sh FILETOSCORE
        python3 score.py -i ${MODELDIR}/train.clean.${TRG}.out -o ${RESDIR}/forward.train.score
        python3 score.py -i ${MODELDIR}/test.clean.${TRG}.out -o ${RESDIR}/forward.test.score

        # 5. Train reversed system to use for backtranslating data
        #    train.sh SYSTEM DATADIR SOURCE TARGET OUTPUTDIR
        REVMODELDIR=${TRG}'-'${SRC}'-'${S}
        train.sh $S $LANGPAIR $TRG $SRC ${REVMODELDIR}

        # 6. Translate with the reverse system
        #    translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        translate.sh $S ${REVMODELDIR} $TRG $SRC train

        # 7. Copy backtranslated data and original data to new folder
        BACKMODELDIR=${SRC}'-'${TRG}'-'${S}'-BACK'
        mkdir ${BACKMODELDIR} -p
        cp ${REVMODELDIR}/train.clean.${TRG}.out ${BACKMODELDIR}/train.clean.${TRG}
        cp ${REVMODELDIR}/test.clean.${TRG}.out ${BACKMODELDIR}/test.clean.${TRG}
        cp ${REVMODELDIR}/dev.clean.${TRG}.out ${BACKMODELDIR}/dev.clean.${TRG}

        # 8. Train system with backtranslated data
        train.sh $S ${BACKMODELDIR} $SRC $TRG

        # 9. Translate text with the backtranslated models
        #    translate.sh SYSTEM DIRWITHMODELS SOURCE TARGET INPUTEXTENSION
        translate.sh $S ${BACKMODELDIR} train
        translate.sh $S ${BACKMODELDIR} test

        # 10. Score the translated text with the backtranslated model
        #     score.sh FILETOSCORE
        python3 score.py -i ${BACKMODELDIR}/train.clean.${TRG}.out -o ${RESDIR}/backward.train.score
        python3 score.py -i ${BACKMODELDIR}/test.clean.${TRG}.out -o ${RESDIR}/backward.test.score
    done
done

