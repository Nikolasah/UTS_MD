import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from feature_engine.outliers import ArbitraryOutlierCapper
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle as pkl

class loanclassModel:
    def __init__(self, filepath):
        self.filepath = ("Dataset_A_loan.csv")
        self.df = pd.read_csv(filepath)
        self.df_encoded = None
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.numerical_cols = [
            'person_age', 'person_emp_exp', 'person_income', 'loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
            'credit_score'
        ]

    def clean_data(self):
        self.df['person_gender'] = self.df['person_gender'].replace({'Male': 'male', 'fe male': 'female'})
        self.df = self.df.dropna().drop_duplicates()

    def cap_outliers(self):
        capper = ArbitraryOutlierCapper(
            max_capping_dict={
                'person_emp_exp': 55,
                'person_age': 70
            }
        )
        self.df = capper.fit_transform(self.df)

    def encode_data(self):
        df_encoded = self.df.copy()
        lb = LabelBinarizer()
        for col in ['previous_loan_defaults_on_file', 'person_gender']:
            df_encoded[col] = lb.fit_transform(df_encoded[[col]])

        cat_cols = ['person_education', 'person_home_ownership', 'loan_intent']
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

        bool_cols = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

        self.df_encoded = df_encoded

    def split_and_scale(self):
        x = self.df_encoded.drop('loan_status', axis=1)
        y = self.df_encoded['loan_status']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=777)
        self.x_train[self.numerical_cols] = self.scaler.fit_transform(self.x_train[self.numerical_cols])
        self.x_test[self.numerical_cols] = self.scaler.transform(self.x_test[self.numerical_cols])

    def train_models(self):
        self.model = XGBClassifier(n_estimators=100, random_state=77, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(self.x_train, self.y_train)

    def finetune_models(self):
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        xgb_grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='error'),
            param_grid=xgb_param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        xgb_grid_search.fit(self.x_train, self.y_train)
        self.model = xgb_grid_search.best_estimator_
        xgb_best_score = xgb_grid_search.best_score_

        print("Best hyperparameters for XGBoost:", xgb_grid_search.best_params_)
        print("Best AUC score for XGBoost:", xgb_best_score)

    def evaluate_models(self):
        pred = self.model.predict(self.x_test)
        proba = self.model.predict_proba(self.x_test)[:, 1]

        print("XGBoost Classifier")
        print(classification_report(self.y_test, pred))
        print(f"ROC AUC: {roc_auc_score(self.y_test, proba):.4f}")

        ConfusionMatrixDisplay.from_estimator(self.model, self.x_test, self.y_test)
        plt.title("XGBoost Confusion Matrix")
        plt.show()

    def run_all(self):
        self.clean_data()
        self.cap_outliers()
        self.encode_data()
        self.split_and_scale()
        self.train_models()
        self.evaluate_models()
        self.finetune_models()
