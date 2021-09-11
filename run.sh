export TRAINING_DATA="Data"
export TARGET_COLS="SIU_Referral_Flag"
export MODEL=$1
FOLD=0 python3 -m src.train
FOLD=1 python3 -m src.train
FOLD=2 python3 -m src.train
FOLD=3 python3 -m src.train
FOLD=4 python -m src.train