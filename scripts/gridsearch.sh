#!/bin/bash

#  10-130: 494
# 140-260: 494
# 270-380: 456
N_PCS_1_START=10
N_PCS_1_STOP=130

# do not change
#  10-380
N_PCS_2_START=10
N_PCS_2_STOP=380

N_PCS_1_STEP=10
N_PCS_2_STEP=10

# COUNT=0
ARRAY="1-2"
MAX_ARRAY="10"

for (( N_PCS_1=${N_PCS_1_START}; N_PCS_1 <= ${N_PCS_1_STOP}; N_PCS_1 += ${N_PCS_1_STEP} ))
do
    for (( N_PCS_2=${N_PCS_2_START}; N_PCS_2 <= ${N_PCS_2_STOP}; N_PCS_2 += ${N_PCS_2_STEP} ))
    do
        N_PCS="${N_PCS_1} ${N_PCS_2}"
        export N_PCS=${N_PCS}
        mysbatcharray /home/mwang8/ukbb-cca/scripts/slurm_job_python.sh ${ARRAY} ${MAX_ARRAY}
        # /home/mwang8/ukbb-cca/scripts/test.sh
        # COUNT=$((COUNT+1))
    done
done

# echo ${COUNT}
