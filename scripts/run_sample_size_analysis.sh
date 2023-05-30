#!/bin/bash

FNAME_DISPATH_SCRIPT="dispatch_job.sh" # assumes all scripts are in same directory
FNAME_CCA_SCRIPT="cca_sample_size.py"

USAGE="Usage: $0 -t/--tag TAG --stratify/--no-stratify --null-model/--no-null-model --scipy-procrustes|--old-procrustes -s/--sample-sizes MIN MAX TOTAL -b/--bootstrap-repetitions MIN MAX TOTAL --min MIN_SAMPLE_SIZE --max MAX_SAMPLE_SIZE [--match-val/--no-match-val] -p/--pcs N PC1 PC2 [...]"

# get path to slurm script directory
if [ -z "${SLURM_JOB_ID}" ]
then
    FPATH_CURRENT="$0"
else
    # check the original location through scontrol and $SLURM_JOB_ID
    FPATH_CURRENT=`scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}' | awk '{print $1}'`
fi
FPATH_CURRENT=`realpath ${FPATH_CURRENT}`
DPATH_CURRENT=`dirname ${FPATH_CURRENT}`
FPATH_DISPATCH_SCRIPT="${DPATH_CURRENT}/${FNAME_DISPATH_SCRIPT}"
FPATH_CCA_SCRIPT="${DPATH_CURRENT}/${FNAME_CCA_SCRIPT}"

if [ "$#" -lt 14 ]
then
    echo $USAGE
    exit 1
fi

MATCH_VAL_STR='--no-match-val'

# parse args
while [[ "$#" -gt 0 ]]
do
    case $1 in
        --min)
            SAMPLE_SIZE_MIN="$2"
            shift 2
            ;;
        --max)
            SAMPLE_SIZE_MAX="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --stratify|--no-stratify)
            STRATIFY="$1"
            shift
            ;;
        --null-model|--no-null-model)
            NULL_MODEL="$1"
            shift
            ;;
        --scipy-procrustes|--old-procrustes)
            SCIPY_PROCRUSTES="$1"
            shift
            ;;
        --match-val)
            MATCH_VAL_STR='--match-val'
            shift
            ;;
        --no-match-val)
            MATCH_VAL_STR='--no-match-val'
            shift
            ;;
        -s|--sample_sizes)
            I_SAMPLE_SIZE_MIN="$2"
            I_SAMPLE_SIZE_MAX="$3"
            N_SAMPLE_SIZES="$4"
            shift 4
            ;;
        -b|--bootstrap-repetitions)
            I_BOOTSTRAP_REPETITION_MIN="$2"
            I_BOOTSTRAP_REPETITION_MAX="$3"
            N_BOOTSTRAP_REPETITIONS="$4"
            shift 4
            ;;
        -p|--pcs)
            n="$2" # number of args to take
            shift 2
            for (( i=0; i < ${n}; i += 1 ))
            do
                N_PC="$1"
                if [ -z "${N_PCS}" ]
                then
                    N_PCS="${N_PC}"
                else
                    N_PCS="${N_PCS} ${N_PC}"
                fi
                shift
            done
            ;;
        *)
            echo "Invalid input argument: $1"
            echo $USAGE
            exit 1
            ;;
    esac
done

if [ -z $FPATH_DISPATCH_SCRIPT ]
then
    echo "Script not found: ${FPATH_DISPATCH_SCRIPT}"
    exit 1
fi

if [ $I_BOOTSTRAP_REPETITION_MAX -gt $N_BOOTSTRAP_REPETITIONS ]
then
    echo "Invalid I_BOOTSTRAP_REPETITION_MAX: $I_BOOTSTRAP_REPETITION_MAX (cannot be greater than ${N_BOOTSTRAP_REPETITIONS})"
    exit 2
fi

if [ $I_SAMPLE_SIZE_MAX -gt $N_SAMPLE_SIZES ]
then
    echo "Invalid I_BOOTSTRAP_REPETITION_MAX: $I_BOOTSTRAP_REPETITION_MAX (cannot be greater than ${N_BOOTSTRAP_REPETITIONS})"
    exit 2
fi

if [ -z $SAMPLE_SIZE_MIN ]
then
    SAMPLE_SIZE_MIN_STR=""
else
    SAMPLE_SIZE_MIN_STR="--min ${SAMPLE_SIZE_MIN}"
fi

if [ -z $SAMPLE_SIZE_MAX ]
then
    SAMPLE_SIZE_MAX_STR=""
else
    SAMPLE_SIZE_MAX_STR="--max ${SAMPLE_SIZE_MAX}"
fi

echo '========== START =========='
echo `date`

for (( I_SAMPLE_SIZE=${I_SAMPLE_SIZE_MIN}; I_SAMPLE_SIZE <= ${I_SAMPLE_SIZE_MAX}; I_SAMPLE_SIZE += 1 ))
do
    for (( I_BOOTSTRAP_REPETITION=${I_BOOTSTRAP_REPETITION_MIN}; I_BOOTSTRAP_REPETITION <= ${I_BOOTSTRAP_REPETITION_MAX}; I_BOOTSTRAP_REPETITION += 1 ))
    do
        if [ "${N_PCS}" = "100 100" ]
        then
            if [ $I_SAMPLE_SIZE -le 10 ]
            then
                DISPATCH_MEMORY="30G" #"20G"
                DISPATCH_TIME="0:30:00"
                # DISPATCH_MEMORY="5G"
                # DISPATCH_TIME="0:15:00"
            elif [ $I_SAMPLE_SIZE -le 20 ]
            then
                DISPATCH_MEMORY="30G"
                DISPATCH_TIME="0:30:00" #"0:45:00"
                # DISPATCH_MEMORY="5G"
                # DISPATCH_TIME="0:15:00"
            else
                DISPATCH_MEMORY="40G"
                DISPATCH_TIME="0:30:00" #"1:30:00"
                # DISPATCH_TIME="0:55:00"
            fi
        elif [ "${N_PCS}" = "300 300" ]
        then
            if [ $I_SAMPLE_SIZE -le 10 ]
            then
                DISPATCH_MEMORY="30G"
                DISPATCH_TIME="0:40:00"
            elif [ $I_SAMPLE_SIZE -le 20 ]
            then
                DISPATCH_MEMORY="30G"
                DISPATCH_TIME="0:50:00"
            else
                DISPATCH_MEMORY="40G"
                DISPATCH_TIME="1:10:00"
            fi
        elif [ "${N_PCS}" = "5 5" ]
        then
            DISPATCH_MEMORY="10G"
            DISPATCH_TIME="0:30:00"
        elif [ "${N_PCS}" = "20 20" ]
        then
            DISPATCH_MEMORY="15G"
            DISPATCH_TIME="0:30:00"
        elif [ "${N_PCS}" = "50 50" ]
        then
            DISPATCH_MEMORY="30G"
            DISPATCH_TIME="0:30:00"
        else
            echo "No memory/time settings for PCs=${N_PCS}. Using defaults"
            DISPATCH_MEMORY="40G"
            DISPATCH_TIME="0:45:00"
        fi

        SUBDIRS_LOG="PCs_${N_PCS// /_}/${TAG}/i_sample_size_${I_SAMPLE_SIZE}"
        
        COMMAND="
            ${FPATH_DISPATCH_SCRIPT} ${FPATH_CCA_SCRIPT} \
            -m ${DISPATCH_MEMORY} -t ${DISPATCH_TIME} \
            --tag ${TAG} ${STRATIFY} ${NULL_MODEL} ${SCIPY_PROCRUSTES} -d ${SUBDIRS_LOG} ${N_SAMPLE_SIZES} \
            ${N_BOOTSTRAP_REPETITIONS} ${I_SAMPLE_SIZE} ${I_BOOTSTRAP_REPETITION} \
            ${N_PCS} ${SAMPLE_SIZE_MIN_STR} ${SAMPLE_SIZE_MAX_STR} ${MATCH_VAL_STR} \
        "
        echo ${COMMAND}
        eval ${COMMAND}
    done
done

echo '========== DONE =========='
echo `date`
