from flask import Flask
from flask import render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import joblib
import config

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///params.db"
db = SQLAlchemy(app)

# カラムの取り出し
df = pd.read_csv('data/BostonHousing.csv',encoding='shift-jis')
# 目的変数
y = df[config.TARGET]
# X = df.drop(y.columns, axis=1)
X = df[config.COLUMNS]
col_names = X.columns

def predict(parameters):
    model = joblib.load('model/lgb_model.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone("Asia/Tokyo")))

@app.route("/", methods=["GET", "POST"])
def index():
    # /の場合はindex.htmlを表示させる
    return render_template("index.html")

@app.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        # POST(value送信)の場合は推論して/にredirectする
        pred_list = []
        for col in col_names:
            # 入力フォームに値が入力されていることを確認
            if request.form.get(col) == "":
                return redirect("/create")
            pred_value = request.form.get(col)
            pred_list.append(pred_value)
        x = np.array(pred_list)
        pred = predict(x)

        # 予測をしたのでDBを更新する
        post = Post()  # インスタンス化
        post.value = pred
        post.created_at = datetime.now(pytz.timezone("Asia/Tokyo"))
        db.session.add(post)
        db.session.commit()
        return redirect("/")
    else:
        # GET(画面表示)の場合はcreate.htmlを表示するだけ
        return render_template("/create.html", col_names=col_names)

@app.route('/<int:id>/delete', methods=["GET"])
def delete(id):
    # 削除するidを取得してDBから削除する
    post = Post.query.get(id)
    db.session.delete(post)
    db.session.commit()
    # 削除後は/resultsにルーティングする
    return redirect("/results")

@app.route("/results")
def results():
    # 結果画面はGETしかないのでDBを更新して表示する
    posts = Post.query.all()
    return render_template("/results.html", posts=posts)