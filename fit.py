import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

# データの読み込み
df = pd.read_csv('data/BostonHousing.csv',encoding='shift-jis')
# 目的変数
y = df[["medv"]]
X = df.drop(y.columns, axis=1)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# データの整形
train = lgb.Dataset(X_train, y_train)
valid = lgb.Dataset(X_test, y_test)

#モデルパラメータの設定
params = {'metric' : 'rmse'}

# モデル学習
model = lgb.train(params, train)

#モデル予測
pred = model.predict(X_test)
print(pred)

# 学習済みモデルの保存
joblib.dump(model, "model/lgb_model.pkl", compress=True)
