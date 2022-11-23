#!/bin/bash

FNAME_DISPATH_SCRIPT="dispatch_job.sh" # assumes all scripts are in same directory
FNAME_CCA_SCRIPT="cca_sample_size.py"

USAGE="Usage: $0 -s/--sample-sizes MIN MAX TOTAL -b/--bootstrap-repetitions MIN MAX TOTAL -p/--pcs N PC1 PC2 [...]"

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

if [ "$#" -lt 12 ]
then
    echo $USAGE
    exit 1
fi

# parse args
while [[ "$#" -gt 0 ]]
do
    case $1 in
        -s|--sample_sizes)
            I_SAMPLE_SIZE_MIN="$2"
            I_SAMPLE_SIZE_MAX="$3"
            N_SAMPLE_SIZES="$4"
            shift # past argument
            shift # past value
            shift # past value
            shift # past value
            ;;
        -b|--bootstrap-repetitions)
            I_BOOTSTRAP_REPETITION_MIN="$2"
            I_BOOTSTRAP_REPETITION_MAX="$3"
            N_BOOTSTRAP_REPETITIONS="$4"
            shift # past argument
            shift # past value
            shift # past value
            shift # past value
            ;;
        -p|--pcs)
            n="$2" # number of args to take
            shift # past argument
            shift # past n
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
            elif [ $I_SAMPLE_SIZE -le 20 ]
            then
                DISPATCH_MEMORY="30G"
                DISPATCH_TIME="0:45:00"
            else
                DISPATCH_MEMORY="40G"
                DISPATCH_TIME="0:55:00"
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
        else
            echo "No memory/time settings for PCs=${N_PCS}. Using defaults"
            DISPATCH_MEMORY="40G"
            DISPATCH_TIME="0:45:00"
        fi

        SUBDIRS_LOG="PCs_${N_PCS// /_}/i_sample_size_${I_SAMPLE_SIZE}"
        
        COMMAND="${FPATH_DISPATCH_SCRIPT} ${FPATH_CCA_SCRIPT} -m ${DISPATCH_MEMORY} -t ${DISPATCH_TIME} -d ${SUBDIRS_LOG} ${N_SAMPLE_SIZES} ${N_BOOTSTRAP_REPETITIONS} ${I_SAMPLE_SIZE} ${I_BOOTSTRAP_REPETITION} ${N_PCS}"
        echo ${COMMAND}
        eval ${COMMAND}
    done
done

echo '========== DONE =========='
echo `date`
