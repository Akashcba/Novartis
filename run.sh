export TRAINING_DATA="Akashcba/Novartis/data/Novartis_Clean_Data.csv"
#export MODEL_PATH="/Users/my_mac/Documents/Machine Learning/ML/models"


# - binary_classification
# - multiclass_classification
# - multilabel_classification
# - single_col_regression
# - multi_col_regression
# - holdout_[%Value] => Very usefull in time series data(Make shuffle = False )=> and for large datasets.

# Cross Validation
export PROBLEM_TYPE="binary_classification"
export TARGET_COLS="False_Fold"
export LABEL_DELIMETER=" "
export NUM_FOLDS="5"

# Categorical Encoding ....
# label
# binary
# ohe
export TYPE="ohe"
# Are there NA values in dataset -> False : No NA values
export NA="False"

export MODEL=$1


python -m src.categorical
python -m src.cross_validation

FOLD=0 python -m src.train
FOLD=2 python -m src.train
FOLD=1 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
#FOLD=5 python -m src.train
#FOLD=6 python -m src.train
#python -m src.predict