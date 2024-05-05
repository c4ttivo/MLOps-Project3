import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
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
    X_test = test[feature_set]
    
    Y_train = train['readmitted']
    Y_test = test['readmitted']
    
    model_name = "xgboost"

    print("nums of train/test set: ", len(X_train), len(X_test), len(Y_train), len(Y_test))
        
    model = xgb.XGBClassifier()

    pipeline = Pipeline([
        ('standard_scaler', StandardScaler()), 
        ('pca', PCA()), 
        ('model', model)
    ])

    param_grid = {
        'pca__n_components': [5, 10, 15, 20, 25, 30],
        'model__max_depth': [2, 3, 5, 7, 10],
        'model__n_estimators': [10, 100, 500],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')    
    
    mlflow.xgboost.autolog(log_model_signatures=True, log_input_examples=True,registered_model_name=model_name)
    with mlflow.start_run(run_name="xgboost_run") as run:

        grid.fit(X_train, Y_train)      
    



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
    X_test = test[feature_set]
    
    Y_train = train['readmitted']
    Y_test = test['readmitted']
    
    
    #Pipeline
    pipeline = Pipeline(steps=[,
        ("scaler", StandardScaler(with_mean=False)),
        ("random_forest", RandomForestClassifier())
    ]) 

    param_grid = { 
        "random_forest__max_depth":[5,10,15],
        "random_forest__n_estimators":[100,150,200]
    }

    model_name = "random_forest"
    search_rf = GridSearchCV(pipeline, param_grid, n_jobs=2)

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True,registered_model_name=model_name)
    with mlflow.start_run(run_name="random_forest_run") as run:

        search_rf.fit(X_train, Y_train)    
    

