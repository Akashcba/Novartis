import os
import time
import pandas as pd
from sklearn import preprocessing

from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
#TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
target = os.environ.get("TARGET_COLS")
#id = os.environ.get("ID_COLS")
MODEL = os.environ.get("MODEL")

Fold_Mapping = {
    0 : [1,2,3,4,5,6],
    1 : [0,2,3,4,5,6],
    2 : [0,1,3,4,5,6],
    3 : [0,1,2,4,5,6],
    4 : [0,1,2,3,5,6],
    5 : [0,1,2,3,4,6],
    6 : [0,1,2,3,4,5]
}

if __name__ == "__main__":
    print("\nExecuting the Train Module\n")
    time.sleep(1)
    df = pd.read_csv(f"/content/Novartis/data/{TRAINING_DATA}.csv")
    #df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(Fold_Mapping.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)
    
    ytrain = train_df[target].values
    yvalid = valid_df[target].values

    #train_df = train_df.drop([id, target, "kfold"], axis=1)
    train_df = train_df.drop([target, "kfold"], axis=1)
    #valid_df = valid_df.drop([id, target, "kfold"], axis=1)
    valid_df = valid_df.drop([target, "kfold"], axis=1)

    print("\nTraining Fold : ", FOLD, "\n")

    valid_df = valid_df[train_df.columns]

    # Data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    #preds = clf.predict_proba(valid_df)[:, 1]
    preds = clf.predict(valid_df)#[:, 1]
    ## Only for lasso
    if MODEL == "lasso":
            for i in range(len(preds)):
                if preds[i] >=0.5:
                    preds[i]= 1
                else:
                    preds[i]= 0
            #print(preds)
    time.sleep(10)
    print("\nF1_Score on validation set : ", metrics.f1_score(yvalid, preds))
    print("AUC Score : ",metrics.roc_auc_score(yvalid, preds),"\n")

    ## Storing the data for predict.py
    print("\nTraining Compeleted\n")
    #joblib.dump(clf, f"/Users/my_mac/Documents/Machine Learning/ML/models/{MODEL}_{FOLD}.pkl")