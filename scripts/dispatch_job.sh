#!/bin/bash 

# dotenv should be in the same directory as this file
FNAME_DOTENV=".env"
FNAME_DISPATCH_LOG="dispatch_job.log"
COMMAND_EXT=".py"

DEFAULT_MEMORY="1G"
DEFAULT_TIMELIMIT="0:5:00"
JOB_ID_SYMBOL="%j"

# input validation
if [ $# -lt 1 ]
then
        echo "Usage: $0 FPATH_COMMAND [ARG1 ARG2 ...] [-m/--mem MEMORY] [-t/--time TIMELIMIT]"
        exit 1
fi

# parse input arguments
export FPATH_COMMAND=`realpath $1`
export ARGS=""

# make sure file exists
if [ ! -f ${FPATH_CURRENT} ]
then
        echo "Error: no file ${FPATH_CURRENT}"
        exit 2
fi

shift # skip first argument
while [[ $# -gt 0 ]]
do
        case $1 in
                -m|--mem)
                        DISPATCH_MEMORY="$2"
                        shift # past argument
                        shift # past value
                        ;;
                -t|--time)
                        DISPATCH_TIMELIMIT="$2"
                        shift # past argument
                        shift # past value
                        ;;
                *)
                        if [ -z "${ARGS}" ]
                        then
                                ARGS="$1"
                        else
                                ARGS="${ARGS} $1"
                        fi
                        shift # past argument
                        ;;
        esac
done

# set default memory/time limit if needed
if [ -z "${DISPATCH_MEMORY}" ]
then
        DISPATCH_MEMORY="${DEFAULT_MEMORY}"
fi
if [ -z "${DISPATCH_TIMELIMIT}" ]
then
        DISPATCH_TIMELIMIT="${DEFAULT_TIMELIMIT}"
fi

# get path to slurm script directory
if [ -z "${SLURM_JOB_ID}" ]
then
    FPATH_CURRENT="$0"
else
    # check the original location through scontrol and $SLURM_JOB_ID
    FPATH_CURRENT=`scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}'`
fi
FPATH_CURRENT=`realpath ${FPATH_CURRENT}`
DPATH_CURRENT=`dirname ${FPATH_CURRENT}`

# load environment variables
set -a
source "${DPATH_CURRENT}/${FNAME_DOTENV}"

# set up logs directory (for slurm job log)
BASENAME_COMMAND=`basename ${FPATH_COMMAND} ${COMMAND_EXT}`
ARGS_PROCESSED="${ARGS}"
ARGS_PROCESSED="${ARGS_PROCESSED// /_}" # replace space by underscore
ARGS_PROCESSED="${ARGS_PROCESSED//-}" # remove dashes
ARGS_PROCESSED="${ARGS_PROCESSED//=/_}" # replace equal signs by underscore
DPATH_OUT="${DPATH_LOGS}/${BASENAME_COMMAND}"
mkdir -p $DPATH_OUT

# slurm log file
if [ -z "${ARGS_PROCESSED}" ]
then
        FPATH_OUT="${DPATH_OUT}/slurm_${JOB_ID_SYMBOL}-${BASENAME_COMMAND}.out"
else
        FPATH_OUT="${DPATH_OUT}/slurm_${JOB_ID_SYMBOL}-${BASENAME_COMMAND}-${ARGS_PROCESSED}.out"
fi

# create log for dispatch_job if necessary
FPATH_DISPATCH_LOG="${DPATH_LOGS}/${FNAME_DISPATCH_LOG}"
if [ ! -f "${FPATH_DISPATCH_LOG}" ]
then
        # header
        echo -e "id\tdate\tscript\targs\tmemory\ttime\tlog" > $FPATH_DISPATCH_LOG
fi

COMMAND="sbatch --parsable --output=${FPATH_OUT} --mem=${DISPATCH_MEMORY} --time=${DISPATCH_TIMELIMIT} ${FPATH_TEMPLATE}"

echo "${COMMAND}"
DISPATCH_JOB_ID=`eval "${COMMAND}"`
echo "${DISPATCH_JOB_ID}"

# add row to log file
echo -e "${DISPATCH_JOB_ID}\t`date`\t${FPATH_COMMAND}\t${ARGS}\t${DISPATCH_MEMORY}\t${DISPATCH_TIMELIMIT}\t${FPATH_OUT//${JOB_ID_SYMBOL}/${DISPATCH_JOB_ID}}" >> $FPATH_DISPATCH_LOG
