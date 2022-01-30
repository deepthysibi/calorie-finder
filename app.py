#!python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    calorie = ''
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        age    = float(request.form.get('age'))
        gender = request.form.get('gender')
        activity = request.form.get('activity')
        intensity = request.form.get('intensity')
        duration = float(request.form.get('duration'))
        if gender == "female":
            gender = 0
        else:
            gender = 1
        if activity == "Outdoor Games":
            activity=0
        elif activity == "Running":
            activity =1
        elif activity == "Walking":
            activity =2
        elif activity == "cardi-gym-activities":
            activity =3
        elif activity == "Cycling":
            activity =4
        elif activity == "swimming":
            activity =5
        else:
            activity=6
        if intensity =="light":
            intensity = 0
        elif intensity == "moderate":
            intensity =1
        else:
            intensity = 2
        calorie = calc_calorie(gender,age,height,weight,activity,intensity,duration) 
    return render_template("Calorie.html",
	                        calorie=calorie)    

def calc_calorie(gender,age,height,weight,activity,intensity,duration):
    df=pd.read_excel("calorie_data.xlsx")
    df1=df.copy()
    le=LabelEncoder()
    df1["Gender"]=le.fit_transform(df1["Gender"])
    df1['Activity']=le.fit_transform(df1["Activity"])
    inten_map={"Light":0,"Moderate":1,"Heavy":2}
    df1["Intensity"]=df1["Intensity"].map(lambda x: inten_map.get(x))
    dur_mean=df1["Duration"].mean()
    heart_mean=df1["Heart_Rate"].mean()
    df1=df1.fillna({"Duration":int(dur_mean),"Heart_Rate":int(heart_mean)})
    x=df1.drop(["Sl_no","Heart_Rate","Calories"],axis=1)
    y=df1["Calories"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
    model1=LinearRegression()
    model1.fit(x_train,y_train)
    return model1.predict([[gender,age,height,weight,activity,intensity,duration]])
    #return round((weight / ((height / 100) ** 2)), 2)

if __name__ == '__main__':
    app.run()