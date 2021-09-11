#!/bin/bash
#RED= '\u001B[31m'
#NC='\033[0m' # No Color
#BOLD=$(tput bold)
#NORM=$(tput sgr0) # Normal

uname -a

printf "The Show is going to Start.!$"
echo -e "@Author: Akash Choudhary"
echo " "
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
    echo -e "Performing Categorical Encoding\n"
    python3 -m src.categorical
    ## Perform cross validation
    echo -e "Performing Cross Validation\n"
    python3 -m src.cross_validation
    done
echo "Training the Model : RandomForest"
sh run.sh randomforest
echo "Training the Model : Logistic Regression"
sh run.sh lr
echo "Training the Model : Naive Bayes"
sh run.sh gnb
echo "Training the Model : XGBoost"
sh run.sh xgb
echo "Training the Model : Logistic Regression"
sh run.sh svm
echo "Training the Model : KNN"
sh run.sh knn
echo -e "Game Over!!" # Ends