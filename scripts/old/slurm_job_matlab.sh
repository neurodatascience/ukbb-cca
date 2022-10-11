#!/bin/bash
#SBATCH --account=def-jbpoline
#SBATCH --mem=100G
#SBATCH --time=3:00:00 # hours:minutes:seconds

FNAME_SCRIPT="compute_cca.m"
ARGS=""

DPATH_PROJECT='/home/mwang8/projects/def-jbpoline/mwang8/ukbb-cca'
DPATH_WORKING="${DPATH_PROJECT}/scripts"

COMMAND="matlab -nodisplay -nojvm -singleCompThread -batch 'cd ${DPATH_WORKING}; run ${FNAME_SCRIPT}'"

echo '===== LOADING MODULES ====='
module load matlab/2021a.5

echo "===== STARTING SCRIPT ====="
echo "${COMMAND}"
eval "${COMMAND}"

echo '===== DONE ====='
echo `date`
