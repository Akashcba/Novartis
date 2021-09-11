#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD=$(tput bold)
NORM=$(tput sgr0) # Normal

uname -a

echo -e "${RED}${BOLD}The Show is going to Start.!${NC}${NORM}"
echo -e "${RED}@Author: Akash Choudhary${NC}"
echo " "
sh warm.sh
echo "${RED}${BOLD}Training the Model : RandomForest${NC}${NORM}"
sh run.sh randomforest
echo "${RED}${BOLD}Training the Model : Logistic Regression${NC}${NORM}"
sh run.sh lr
echo "${RED}${BOLD}Training the Model : Naive Bayes${NC}${NORM}"
sh run.sh gnb
echo "${RED}${BOLD}Training the Model : XGBoost${NC}${NORM}"
sh run.sh xgb
echo "${RED}${BOLD}Training the Model : KNN${NC}${NORM}"
sh run.sh knn
echo -e "${RED}${BOLD}Game Over!!${NC}${NORM}" # Ends