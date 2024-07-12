import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import config

def fit_(X, y, param, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # データの整形
    train = lgb.Dataset(X_train, y_train)
    valid = lgb.Dataset(X_test, y_test)
    #モデルパラメータの設定
    param = {'metric' : param}
    # モデル学習
    model = lgb.train(param, train)
    # 学習済みモデルの保存
    joblib.dump(model, f"model/{name}.pkl", compress=True)