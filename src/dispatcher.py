from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


MODELS = {
    'randomforest' : ensemble.RandomForestClassifier(n_estimators=500,n_jobs=1, verbose=0),
    'extratrees' : ensemble.ExtraTreesClassifier(n_estimators=500,n_jobs=1, verbose=0),
    'gbm' : ensemble.GradientBoostingClassifier(n_estimators=500, verbose=0),
    'knn' : KNeighborsClassifier(n_neighbors=5),
    'xgb' : XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=800, objective='binary:logistic', booster='gbtree', verbose=0),
    'gnb':GaussianNB(),
    'lr' : LogisticRegression(solver='liblinear', random_state=0),
    'svm':svm.SVC(),
    'lasso' : Lasso(alpha=0.1)
}
