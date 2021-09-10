export TRAINING_DATA="/content/Novartis/data/Novartis_Clean_Data.csv"
export MODEL_PATH="/content/Novartis/models/"


# - binary_classification
# - multiclass_classification
# - multilabel_classification
# - single_col_regression
# - multi_col_regression
# - holdout_[%Value] => Very usefull in time series data(Make shuffle = False )=> and for large datasets.

# Cross Validation
export PROBLEM_TYPE="binary_classification"
export TARGET_COLS="False_Flag"
export ID_COLS="Policy_ID"
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


#python3 -m src.categorical
#python3 -m src.cross_validation

FOLD=0 python3 -m src.train
FOLD=2 python3 -m src.train
FOLD=1 python3 -m src.train
FOLD=3 python3 -m src.train
FOLD=4 python -m src.train
#FOLD=5 python -m src.train
#FOLD=6 python -m src.train
#python -m src.predict