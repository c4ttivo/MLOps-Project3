import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import xgboost as xgb
import mlflow


# MLFlow configs

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("mlflow_training_tracking")

DATABASE_CLEAN_URI = 'postgresql+psycopg2://cleanusr:cleanpass@cleandb/cleandb'
RANDOM_STATE = 42


def get_training_data():
    engine = create_engine(DATABASE_CLEAN_URI)
    train_df = pd.read_sql('diabetes_training_data', engine)

    return train_df

def get_test_data():
    engine = create_engine(DATABASE_CLEAN_URI)
    test_df = pd.read_sql('diabetes_test_data', engine)

    return test_df


def train_xgboost(): 

    feature_set = ['race', 'gender', 'age',
                   'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                   'time_in_hospital', 'num_lab_procedures',
                   'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'diag_1', 'number_diagnoses',
                   'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                   'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide',
                   'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                   'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin',
                   'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                   'diabetesMed', 'num_med_changed', 'num_med_taken']

    train = get_training_data()
    test = get_test_data()
    
    X_train = train[feature_set]
    X_test = test[feature_set
    
    Y_train = train['readmitted']
    Y_test = test['readmitted']
    
    model_name = "xgboost"
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True,registered_model_name=model_name)

    print("nums of train/test set: ", len(X_train), len(X_test), len(Y_train), len(Y_test))

    #-------- XGboost ---------#
    print('--- XGBoost model ---')
    xg_reg = xgb.XGBClassifier()

    # print("Cross Validation score: ", np.mean(cross_val_score(xg_reg, X_train, Y_train, cv=10)))  # 10-fold 交叉验证
    xg_reg.fit(X_train, Y_train)

    Y_test_predict = xg_reg.predict(X_test)
    acc = accuracy_score(Y_test, Y_test_predict)
    mat = confusion_matrix(Y_test, Y_test_predict)
    f1 = f1_score(Y_test, Y_test_predict, average='weighted')
    print("Accuracy: ", acc)
    print("F1 score: ", f1)
    print("Confusion matrix: \n", mat)
    print('Overall report: \n', classification_report(Y_test, Y_test_predict))



def train_random_forest():

    feature_set = ['race', 'gender', 'age',
                   'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                   'time_in_hospital', 'num_lab_procedures',
                   'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'diag_1', 'number_diagnoses',
                   'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                   'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide',
                   'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                   'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin',
                   'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                   'diabetesMed', 'num_med_changed', 'num_med_taken']

    train = get_training_data()
    test = get_test_data()
    
    X_train = train[feature_set]
    X_test = test[feature_set
    
    Y_train = train['readmitted']
    Y_test = test['readmitted']
    
    model_name = "random_forest"
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True,registered_model_name=model_name)

    print("nums of train/test set: ", len(X_train), len(X_test), len(Y_train), len(Y_test))

    # ------- RF --------#
    print('--- Random-forest model ---')

    forest = RandomForestClassifier(n_estimators=100, max_depth=120, criterion="entropy")
    # print("Cross Validation Score: ", np.mean(cross_val_score(forest, X_train, Y_train, cv=10)))
    forest.fit(X_train, Y_train)

    Y_test_predict = forest.predict(X_test)
    acc = accuracy_score(Y_test, Y_test_predict)
    mat = confusion_matrix(Y_test, Y_test_predict)
    f1 = f1_score(Y_test, Y_test_predict, average='weighted')
    print("Accuracy: ", acc)
    print("F1 score: ", f1)
    print("Confusion matrix: \n", mat)
    print('Overall report: \n', classification_report(Y_test, Y_test_predict))

