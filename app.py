from flask import Flask
from flask import render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///params.db"
db = SQLAlchemy(app)

# カラムの取り出し
df = pd.read_csv('data/BostonHousing.csv',encoding='shift-jis')
# 目的変数
y = df[["medv"]]
X = df.drop(y.columns, axis=1)
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
    if request.method == "GET":
        posts = Post.query.all()
        return render_template("index.html", posts=posts)

@app.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        crim = request.form.get("crim")
        zn = request.form.get("zn")
        indus = request.form.get("indus")
        chas = request.form.get("chas")
        nox = request.form.get("nox")
        rm = request.form.get("rm")
        age = request.form.get("age")
        dis = request.form.get("dis")
        rad = request.form.get("rad")
        tax = request.form.get("tax")
        ptratio = request.form.get("ptratio")
        b = request.form.get("b")
        lstat = request.form.get("lstat")

        x = np.array([crim, zn, indus, chas, nox, rm, 
                    age, dis, rad, tax, ptratio, b, lstat])
        pred = predict(x)

        post = Post(value=pred)
        db.session.add(post)
        db.session.commit()
        return redirect("/")
    else:
        return render_template("/create.html", col_names=col_names)

@app.route('/<int:id>/delete', methods=["GET"])
def delete(id):
    post = Post.query.get(id)
    db.session.delete(post)
    db.session.commit()
    return render_template("/results.html")

@app.route("/results")
def results():
    posts = Post.query.all()
    return render_template("/results.html", posts=posts)