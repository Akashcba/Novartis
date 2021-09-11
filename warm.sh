#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD=$(tput bold)
NORM=$(tput sgr0) # Normal

FILES=""
#FILES="${FILES} Data_Sig"
#FILES="${FILES} Data_Sig_mod"
FILES="${FILES} Data"
for FILE in ${FILES}; do
    export TRAINING_DATA=${FILE};
    export PROBLEM_TYPE="binary_classification"
    export TARGET_COLS="SIU_Referral_Flag"
    export LABEL_DELIMETER=" "
    export NUM_FOLDS="5"
    export TYPE="ohe" # Categorical Encoding ....
    # Are there NA values in dataset -> False : No NA values
    export NA="False"
    ## Encode the data
    echo -e "${RED}Performing Categorical Encoding${NC}\n"
    python3 -m src.categorical
    ## Perform cross validation
    echo -e "${RED}Performing Cross Validation${NC}\n"
    python3 -m src.cross_validation
    done