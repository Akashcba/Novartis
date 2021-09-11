''' @Author : Akash Choudhary
    20BM6JP46
    akashcba2022@email.iimcal.ac.in
'''

import pandas as pd
import os
import joblib
import time
from sklearn import preprocessing

#from colorama import Fore, Style, init
#init()

TRAINING_DATA = os.environ.get("TRAINING_DATA")
#TESTING_DATA = os.environ.get("TEST_DATA")
TYPE = os.environ.get("TYPE")
NA = os.environ.get("NA")
#PATH = os.environ.get("MODEL_PATH")
'''
class Warning(Exception):
    def __init__(self, message):
        super().__init__( Fore.RED + message)
'''


class CategoricalFeatures:
    def __init__(self, dataframe, categorical_features, encoding_type, handle_na=False):
        """
        dataframe: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.dataframe = dataframe
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na=="True" or self.dataframe[self.cat_feats].isnull().values.any():
            print(f"\nThere are Null Values in the DataSet or handle_na is set as True.")
            print("\nNull Values are as follows: \n",self.dataframe[self.cat_feats].isnull().sum())
            #print(f"{Fore.YELLOW}\nPlease handle na values separately or PRESS - Y or y to continue with standard procedure. {Style.RESET_ALL}")
            print("\nPlease handle na values separately or PRESS - Y or y to continue with standard procedure.")
            i=input("> ")
            if i not in ('Y','y'):
                raise Warning("\nNan Values Found.\nExiting ...............")
        
        if self.handle_na:
            for c in self.cat_feats:
                self.dataframe.loc[:, c] = self.dataframe.loc[:, c].astype(str).fillna('-1')
        ## Ouput DataFrame
        self.output_dataframe = self.dataframe.copy(deep=True)
    
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.dataframe[c].values)
            self.output_dataframe.loc[:, c] = lbl.transform(self.dataframe[c].values)
            self.label_encoders[c] = lbl

        #joblib.dump(self.label_encoders,f"{PATH}label_encoder.pkl")
        #TRAINING_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/train_folds.csv"
        # for c,lbl in self.label_encoder.items():
        return self.output_dataframe
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.dataframe[c].values)
            val = lbl.transform(self.dataframe[c].values)
            self.output_dataframe = self.output_dataframe.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_dataframe[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        joblib.dump(self.binary_encoders,f"{PATH}binary_encoder.pkl")
        
        return self.output_dataframe

    def _one_hot(self):
        self.output_dataframe = pd.get_dummies(data=self.dataframe, columns=self.cat_feats, drop_first=True)
        '''
        self.ohe = preprocessing.OneHotEncoder()
        self.ohe.fit(self.dataframe[self.cat_feats].values)
        self.output_dataframe = self.ohe.transform(self.dataframe[self.cat_feats].values)
        joblib.dump(self.ohe,f"{PATH}ohe_.pkl")
        '''
        return self.output_dataframe

    def fit_transform(self):
        if self.enc_type == "label":
            print("Running label_encoding .......")
            return self._label_encoding()
        elif self.enc_type == "binary":
            print("Running Binarization .......")
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Warning("Encoding type not understood")
    


if __name__ == "__main__":
    time.sleep(7)
    train = pd.read_csv(f"/content/Novartis/data/{TRAINING_DATA}.csv")

    train_len = len(train)

    cat_cols = train.dtypes=='object'
    cat_cols = list(cat_cols[cat_cols].index)
    cat_cols = [c for c in cat_cols if c not in ("Policy_ID", "False_Flag")]
    if TYPE=="ohe":
        print("Running One Hot Encoding ........")
    print("\nShape of Train input : ", train.shape)
    cat_feats = CategoricalFeatures(dataframe=train,
                                    categorical_features=cat_cols,
                                    encoding_type=TYPE,
                                    handle_na = NA)
    df = pd.DataFrame(cat_feats.fit_transform())
    #print("Type is ")
    #print(type(df))
    train = df.iloc[ : train_len ,]
    #test = df.iloc[train_len :, ]
    ## Storing the Files .....
    print("Shape of Train After Categorical Encoding : ",train.shape)
    #print("\nShape of Test : ",test.shape)
    #print("Train Columns : \n", train.columns)
    #print("Test Columns : \n", test.columns)
    train.to_csv(f"/content/Novartis/data/{TRAINING_DATA}.csv", index=False)
    #test.to_csv(f"{TESTING_DATA[:-4]}.csv", index=False)
    #joblib.dump(train.columns, f"{PATH}columns.pkl")
    print("\nFile Successfully Modified and saved...\n")