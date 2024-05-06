import torch
from torch.utils.data import TensorDataset
import os
import numpy as np
import pandas as pd



class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()



def load_adult_data(path="../datasets/adult", sensitive_attribute="sex"):
    column_names = ["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native-country","target"]

    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "native-country", "education"]
    features_to_drop = ["fnlwgt"]

    df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python")
    df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python", skiprows=1)

    df = pd.concat([df_train, df_test])
    df.drop(columns=features_to_drop, inplace=True)
    df.dropna(inplace=True)

    # df = pd.get_dummies(df, columns=categorical_features)

    if sensitive_attribute == "race":
        df = df[df["race"].isin(["White", "Black"])]
        s = df[sensitive_attribute][df["race"].isin(["White", "Black"])]
        s = (s == "White").astype(int).to_frame()
        categorical_features.append( "sex" )

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append( "race" )

    df["target"] = df["target"].replace({"<=50K.": 0, ">50K.": 1, ">50K": 1, "<=50K": 0})
    y = df["target"]

    X = df.drop(columns=["target", sensitive_attribute])
    # X = pd.get_dummies(X, columns=categorical_features)
    X[categorical_features] = X[categorical_features].astype("string")

    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s



def load_bank_marketing_data(path="../datasets/bank", sensitive_attribute="age"):
    df = pd.read_csv(os.path.join(path, "bank-additional-full.csv"), sep=";")
    categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]

    
    df["y"] = df["y"].replace({"yes": 1, "no": 0})
    y = df["y"].to_frame()
    s = df[sensitive_attribute]
    s = (s >= 25).astype(int).to_frame()

    X = df.drop(columns=["y", "age"])

    X[categorical_features] = X[categorical_features].astype("string")

    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s



# We set ethnicity and age as the sensitive attribute and the target label, respectively. 
def load_celeba_data(path="../datasets/celeba", sensitive_attribute="race"):
    # chagne the personal_status name to sex race and the target label to age

    df = pd.read_csv( os.path.join(path, "celeba.csv"), na_values="NA", index_col=None, sep=",", header=0)
    df['pixels']= df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x/255)
    df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(x, (3, 48,48)))

    X = df['pixels'].to_frame()
    
    df["Gender"] = df["Male"]
    attr = df[ ["Smiling", "Wavy_Hair", "Attractive", "Male", "Young"  ]]

    return X, attr
