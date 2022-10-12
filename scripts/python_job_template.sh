#!/bin/bash
#SBATCH --account=def-jbpoline

if [ -z "${FPATH_COMMAND}" ]
then
    echo "ERROR: No script given to template"
    exit 1
fi

COMMAND="python -u ${FPATH_COMMAND} ${ARGS}"

# to load/install
MODULES=( "python/3.9.6" )
# downloaded using "pip download --no-deps NAME"
DOWNLOADED_PACKAGES=( \
    "tensorly-0.7.0-py3-none-any.whl" \
    "mvlearn-0.4.1-py3-none-any.whl" \
    "cca_zoo-1.10.16-py3-none-any.whl" \
)

echo '========== START =========='
echo `date`
echo "${COMMAND}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE}"
TIMELIMIT="$(squeue -j ${SLURM_JOB_ID} -h --Format TimeLimit | xargs)"
echo "TIMELIMIT=${TIMELIMIT}"

echo '========== LOADING MODULES =========='
echo `date`
for MODULE in "${MODULES[@]}"
do
    echo "${MODULE}"
    module load "${MODULE}"
done

echo '========== SETTING UP VIRTUAL ENVIRONMENT =========='
echo `date`

virtualenv --no-download "${SLURM_TMPDIR}/env"
source "$SLURM_TMPDIR/env/bin/activate"

# upgrade pip first
pip install --no-index --upgrade pip

# install available dependencies
pip install --no-index -r "${FPATH_REQUIREMENTS}"

# install pre-downloaded packages (not available as wheels)
for DOWNLOADED_PACKAGE in "${DOWNLOADED_PACKAGES[@]}"
do
    echo "----- Installing ${DOWNLOADED_PACKAGE} -----"
    pip install --no-index "${DPATH_TOOLBOX}/${DOWNLOADED_PACKAGE}"
done

# install local src package
pip install --no-index -e "${DPATH_PROJECT}"

echo '========== STARTING SCRIPT =========='
echo `date`
echo "${COMMAND}"
eval "${COMMAND}"

echo '========== DONE =========='
echo `date`

