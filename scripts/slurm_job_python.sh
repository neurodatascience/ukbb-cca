#!/bin/bash
#SBATCH --account=def-jbpoline
#SBATCH --mem=1G
#SBATCH --time=0:30:00

DPATH_PROJECT="/home/mwang8/projects/def-jbpoline/mwang8/ukbb-cca"

# sometimes these need to be commented out:
# bootstrap_check.py: N_PCS and I_SPLIT
# gridsearch.sh/gridsearch_check.py: N_PCS only
# N_PCS="300 300" #"230 310" "240 370"
# I_SPLIT="7"

# FNAME_SCRIPT="generate_bootstrap_samples.py" # 1G 0:5:00
# ARGS=""

# N_PCS="10 10"
# N_PCS="50 50"
N_PCS="100 100"

FNAME_SCRIPT="plot_sample_size_results.py" # 1G 0:5:00 should be plenty
ARGS="${N_PCS}"

# # 30 sample sizes 100 100: 20G 0:30:00 good for at least until i_sample_size=20/30
# #                          40G 0:45:00 worked for i_sample_size=30/30 (sacct_job 10G 0:40:00)
# # 30 sample sizes 50 50: 20G 0:30:00 good for at least until i_sample_size=20/30
# #                        40G 0:45:00 worked for i_sample_size=30/30 (sacct_job 10G 0:40:00)
# FNAME_SCRIPT="cca_sample_size.py" # works with 20G 0:30:00 for 50 50 PCs (for up to i_sample_size=15/20)
# ARGS="30 100 ${I_SAMPLE_SIZE} ${I_BOOTSTRAP_REPETITION} ${N_PCS}" # n_sample_sizes n_bootstrap_repetitions i_sample_size i_bootstrap_repetition n_PCs1 n_PCs2
# # ARGS="30 100 30 1 ${N_PCS}" # n_sample_sizes n_bootstrap_repetitions i_sample_size i_bootstrap_repetition n_PCs1 n_PCs2

# FNAME_SCRIPT="split_data.py" # 2G 0:5:00
# ARGS="10"

# FNAME_SCRIPT="preprocess_data.py" # 3G 0:5:00 for 350 350
# ARGS="${N_PCS}"

# FNAME_SCRIPT="bootstrap_cca.py" # 2G 1:00:00 for 100 100 (1000 iterations), 3G 1:00:00 for 350 350 (250 iterations)
# ARGS="${N_PCS}"

# FNAME_SCRIPT="train_cca_cv.py" # 10G 0:10:00 for 387 387
# # # SLURM_ARRAY_TASK_ID="1"
# # # ARGS="50reps ${SLURM_ARRAY_TASK_ID} ${I_SPLIT} ${N_PCS}" # NORMAL
# ARGS="gs ${SLURM_ARRAY_TASK_ID} ${I_SPLIT} ${N_PCS}" # GRIDSEARCH (commend out N_PCS definition above)
# # ARGS="bootstrap${SLURM_ARRAY_TASK_ID} 1 ${I_SPLIT} ${N_PCS}" # BOOTSTRAP
# # ARGS="bootstrap${SLURM_ARRAY_TASK_ID} ${I_REP} ${I_SPLIT} ${N_PCS}" # BOOTSTRAP CHECK

# # for 50 reps, only need about 10G, 0:10:00 (for 350 350)
# # for 1 rep: 500M 0:5:00 (300 300)
# FNAME_SCRIPT="combine_cv_results.py" # 20G 0:20:00 for 25 25, 50G 0:20:00 for 40 40, 70G 0:40:00 for 100 100
# # ARGS="cv_300_300_50reps_split${I_SPLIT}" # NORMAL
# ARGS="cv_300_300_bootstrap${SLURM_ARRAY_TASK_ID}_split${I_SPLIT}" # BOOTSTRAP

# # for 50 reps (IF WITH DECONF: ADD ~80G?)
# #  25  25: 25G, 1:45:00 works
# # 100 100: 75G, 1:45:00 works
# # 200 200: 150G, 2:00:00 works
# # 300 300: 200G, 2:10:00 works --> works for 240 370 (failed once due to time) and 230 310
# # 350 350: 250G, 2:15:00 works
# # 387 387: 300G, 2:15:00 works
# # for 1 rep
# # 300 300: 10G, 0:20:00 (works for most cases)
# # with deconf
# # 300 300: 500G, 4:00:00 (worked, took 2:37min, >285G) (WITHOUT PCS)
# # --> if with PCs: 400G, 4:00:00 works
# FNAME_SCRIPT="get_central_tendencies.py"
# ARGS="cv_300_300_50reps_split${I_SPLIT}"
# ARGS="cv_300_300_bootstrap${SLURM_ARRAY_TASK_ID}_split${I_SPLIT}" # BOOTSTRAP

# # for 50 reps
# # 25 25: 3G, 5:00
# # 387 387: 15G, 0:45:00
# FNAME_SCRIPT="plot_cca_results.py"
# ARGS="cv_300_300_50reps_split${I_SPLIT}"

# # 2G, 0:30:00
# FNAME_SCRIPT="combine_gridsearch_results.py"
# ARGS="${DPATH_PROJECT}/scratch/cca_gridsearch"

# # # 30G, 0:10:00
# FNAME_SCRIPT="merge_central_tendencies.py"
# ARGS="${DPATH_PROJECT}/results/cca/central_tendencies/cv_300_300_50reps ${DPATH_PROJECT}/results/cca_bootstrap/central_tendencies/"

DPATH_TOOLBOX="${DPATH_PROJECT}/toolbox"
DPATH_SCRIPTS="${DPATH_PROJECT}/scripts"
FPATH_SCRIPT="${DPATH_SCRIPTS}/${FNAME_SCRIPT}"
FPATH_REQUIREMENTS="${DPATH_PROJECT}/requirements.txt"

COMMAND="python -u ${FPATH_SCRIPT} ${ARGS}"

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

echo '========== LOADING MODULES =========='
for MODULE in "${MODULES[@]}"
do
    echo "${MODULE}"
    module load "${MODULE}"
done

echo '========== SETTING UP VIRTUAL ENVIRONMENT =========='

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

echo '========== START SCRIPT =========='
echo "${COMMAND}"
eval "${COMMAND}"

echo '========== DONE =========='
echo `date`

