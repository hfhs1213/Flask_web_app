from flask import Flask
from flask import render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import joblib
import config
import refit

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///params.db"
app.secret_key = 'key'
db = SQLAlchemy(app)

# カラムの取り出し
df = pd.read_csv('data/BostonHousing.csv',encoding='shift-jis')
# 目的変数
y = df[config.TARGET]
# X = df.drop(y.columns, axis=1)
X = df[config.COLUMNS]
col_names = X.columns

def predict(parameters, model_name):
    model = joblib.load(f'model/{model_name}.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

class Post(db.Model):
    __tablename__ = 'post'
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone("Asia/Tokyo")))
    model_name = db.Column(db.String, nullable=False)

class ModelResult(db.Model):
    __tablename__ = 'model_result'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone("Asia/Tokyo")))
    # 精度も保存したい

@app.route("/", methods=["GET", "POST"])
def index():
    # /の場合はindex.htmlを表示させる
    return render_template("index.html")

@app.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        model_name = request.form.get("name")
        # POST(value送信)の場合は推論して/にredirectする
        pred_list = []
        for col in col_names:
            # 入力フォームに値が入力されていることを確認
            if request.form.get(col) == "":
                return redirect("/create")
            pred_value = request.form.get(col)
            pred_list.append(pred_value)
        x = np.array(pred_list)
        pred = predict(x, model_name)

        # 予測をしたのでDBを更新する
        post = Post()  # インスタンス化
        post.value = pred
        post.created_at = datetime.now(pytz.timezone("Asia/Tokyo"))
        post.model_name = model_name
        db.session.add(post)
        db.session.commit()
        return redirect("/")
    else:
        # GET(画面表示)の場合はcreate.htmlを表示するだけ
        model_result = ModelResult.query.all()
        # import pdb; pdb.set_trace()
        models = [model.model_name for model in model_result]
        models.insert(0, "")  # 初期値を空白にしたい
        return render_template("/create.html", col_names=col_names, models=models)

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

@app.route("/fit", methods=["GET", "POST"])
def fit():
    if request.method == "GET":
        # /の場合はfit.htmlを表示させる
        return render_template("fit.html")
    else:
        param = request.form.get('radio')
        model_name = request.form.get('model_name')
        if model_name == "":
            flash("モデル名を入力してください！")
            return render_template("fit.html")
        refit.fit_(X, y, param, model_name)
        flash(f"モデル名:[{model_name}]の学習が完了しました。")
        # DBに登録
        model_result = ModelResult()  # インスタンス化
        model_result.model_name = model_name
        model_result.created_at = datetime.now(pytz.timezone("Asia/Tokyo"))
        db.session.add(model_result)
        db.session.commit()
        # 保存しただけなので表示したい
        return render_template("fit.html")
    
@app.route("/model_results", methods=["GET", "POST"])
def model_results():
    models = ModelResult.query.all()
    # import pdb; pdb.set_trace()
    return render_template("model_results.html", models=models)

@app.route('/<int:id>/delete_model', methods=["GET"])
def delete_model(id):
    # 削除するidを取得してDBから削除する
    model = ModelResult.query.get(id)
    db.session.delete(model)
    db.session.commit()
    # 削除後は/resultsにルーティングする
    return redirect("/model_results")