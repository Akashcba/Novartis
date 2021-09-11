uname -a
declare -a DATA_ARRAY
DATA_ARRAY = ("/content/Novartis/data/Data_Sig.csv" "/content/Novartis/data/Data_Sig_mod.csv" "/content/Novartis/data/Data.csv")

for data in DATA_ARRAY
do
echo "Data Source is $data"
export TRAINING_DATA=data
#export TRAINING_DATA="/content/Novartis/data/Novartis_Clean_Data.csv"
# Significant features Selected by Chi Sq test and VIF Test
#export TRAINING_DATA="/content/Novartis/data/Data_Sig.csv"
# Significant features along with engineered features
#export TRAINING_DATA="/content/Novartis/data/Data_Sig_mod.csv"
# Full data
#export TRAINING_DATA="/content/Novartis/data/Data.csv"

export MODEL_PATH="/content/Novartis/models/"


# - binary_classification
# - multiclass_classification
# - multilabel_classification
# - single_col_regression
# - multi_col_regression
# - holdout_[%Value] => Very usefull in time series data(Make shuffle = False )=> and for large datasets.

# Cross Validation
export PROBLEM_TYPE="binary_classification"
export TARGET_COLS="SIU_Referral_Flag"
#export ID_COLS="Policy_ID"
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

echo "Model is $MODEL"

## Encode the data
echo "Performing Categorical Encoding"
python3 -m src.categorical

## Perform cross validation
echo "Creating Folds for Cross Validation"
python3 -m src.cross_validation

echo "Training the Model : $MODEL"

FOLD=0 python3 -m src.train
FOLD=1 python3 -m src.train
FOLD=2 python3 -m src.train
FOLD=3 python3 -m src.train
FOLD=4 python -m src.train

done
#FOLD=5 python -m src.train
#FOLD=6 python -m src.train
#python -m src.predict