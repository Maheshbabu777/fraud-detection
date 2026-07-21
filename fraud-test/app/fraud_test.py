import json
import warnings
import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import pickle as pkl
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

artifact_path="../artifacts"
data_path="../../ieee-fraud-detection/dataset-org"

class FraudTest:
    def __init__(self):
        self.imputations=json.load(open(artifact_path+ "/imputation_values.json", 'r'))
        self.X_transactions=pd.read_csv(open(data_path+"/test_transaction.csv",'r'))
        self.X_identity=pd.read_csv(open(data_path+"/test_identity.csv",'r'))
        self.colmns_to_drop=json.load(open(artifact_path+"/dropped_columns.json",'r'))
        self.X_test=None
        self.threshold=json.load(open(artifact_path+"/threshold.json",'r'))['threshold']
        self.feature_names=json.load(open(artifact_path+"/feature_names.json",'r'))
        self.email_freq=json.load(open(artifact_path+"/email_frq_lookup.json",'r'))
        self.amount_mean=json.load(open(artifact_path+"/amt_zscore_stats.json",'r'))['mean']   
        self.amount_std=json.load(open(artifact_path+"/amt_zscore_stats.json",'r'))['std']
        self.amt_bin_edges=np.load((artifact_path+"/amt_bin_edges.npy"))
        self.amt_sorted=np.load((artifact_path+"/amt_sorted_train.npy"))
        self.lgbm_model=lgb.Booster(model_file=str(artifact_path+"/lgbm_model.lgb"))
        self.scaler=joblib.load(artifact_path+"/scaler.pkl")
        self.label_encoders=joblib.load(artifact_path+"/label_encoders.pkl")
        self.predictions=None

    def merge_data(self):
        self.X_test=pd.merge(self.X_transactions,self.X_identity,how='left',on='TransactionID')
        self.X_test.columns = self.X_test.columns.str.replace('-', '_')

    def drop_columns(self):
        cols_to_drop=[col for col in self.colmns_to_drop if col in self.X_test.columns]
        self.X_test.drop(columns=cols_to_drop,inplace=True)

    def missing(self):
        missing_from_imputation_dict = [c for c in self.X_test.columns if c not in self.imputations]
        print(len(missing_from_imputation_dict))
        print(missing_from_imputation_dict[:30])

    def impute_missing_vals(self):
        for col, val in self.imputations.items():
            if col in self.X_test.columns:
                self.X_test[col].fillna(val,inplace=True)
    
    def cols_with_missing_data(self):
        missing_cols=self.X_test.isna().sum()
        missing_cols=missing_cols[missing_cols>0]
        print(f"Columns with missing data: {len(missing_cols)}")
        print(missing_cols)

    def add_new_features(self):
        second_in_day=24*60*60
        self.X_test["txn_hour"]=(self.X_test["TransactionDT"]*3600)%24
        self.X_test["txn_day"]=(self.X_test["TransactionDT"]/second_in_day).astype(int)
        self.X_test["txn_weekday"]=(self.X_test["txn_day"]%7)
        self.X_test["is_weekend"]=(self.X_test["txn_weekday"].isin([5,6]).astype(int))
        self.X_test["is_night"]=(self.X_test["txn_hour"].between(0,6).astype(int))

        self.X_test["amt_log"]=np.log1p(self.X_test["TransactionAmt"])
        self.X_test["amt_bin"]=pd.cut(self.X_test["TransactionAmt"],bins=self.amt_bin_edges,labels=False,include_lowest=True)
        self.X_test["amt_percent"]=self.X_test["TransactionAmt"].apply(lambda x: np.searchsorted(self.amt_sorted, x) / len(self.amt_sorted))
        self.X_test["amt_zscore"]=(self.X_test["TransactionAmt"]-self.amount_mean)/self.amount_std

        self.X_test["is_same_mail"]=(self.X_test["P_emaildomain"]==self.X_test["R_emaildomain"]).astype(int)
        self.X_test["email_rarity"]=self.X_test["P_emaildomain"].map(self.email_freq).fillna(0)
        self.X_test["email_is_rare"]=(self.X_test["email_rarity"]<0.01).astype(int)

    def encode_features(self):
        for col,le in self.label_encoders.items():
            if col in self.X_test.columns:
                known = set(le.classes_)
                self.X_test[col]=self.X_test[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                self.X_test[col]=le.transform(self.X_test[col].astype(str))

    def scale_features(self):
        self.X_test=self.X_test[self.feature_names]
        self.X_test=self.scaler.transform(self.X_test)

    def predict(self):
        pred=self.lgbm_model.predict(self.X_test)
        self.predictions=(pred>self.threshold).astype(int)
        print(f"Predictions: {self.predictions.sum()} frauds detected out of {len(self.predictions)} transactions.")
    
    def shap_analysis(self, sample_size=500):
        sample = self.X_test[:sample_size]
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, feature_names=self.feature_names, show=False)
        plt.savefig(artifact_path+"/shap_summary_plot.png", bbox_inches='tight')

test=FraudTest()
test.merge_data()
test.drop_columns()
test.missing()
test.impute_missing_vals()
test.cols_with_missing_data()
test.add_new_features()
test.encode_features()
test.scale_features()
test.predict()
test.shap_analysis()
