from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load("models/model.pkl")


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")


@app.route('/predict', methods=['POST'])
def predict():

    fullname = request.form['fullname']
    gender_text = request.form['gender']
    car_text = request.form['car']
    property_text = request.form['property']

    children = int(request.form['children'])
    income = float(request.form['income'])

    income_type = request.form['income_type']
    education = request.form['education']
    family_status = request.form['family_status']
    housing = request.form['housing']

    employment_days = int(request.form['employment'])
    occupation = request.form['occupation']
    family_members = int(request.form['family_members'])

    dob = request.form['dob']

    mobile = request.form['mobile']
    email = request.form['email']
    pan = request.form['pan']
    aadhaar = request.form['aadhaar']
    address = request.form['address']
    city = request.form['city']
    state = request.form['state']
    pincode = request.form['pincode']

    credit_score = int(request.form['credit_score'])
    active_loans = int(request.form['active_loans'])
    monthly_emi = float(request.form['monthly_emi'])
    credit_card_count = int(request.form['credit_card_count'])
    missed_payments = int(request.form['missed_payments'])
    default_history = int(request.form['default_history'])
    work_phone = int(request.form['work_phone'])


    birth_date = datetime.strptime(dob, "%Y-%m-%d")
    age = datetime.today().year - birth_date.year


    gender = 1 if gender_text == "Male" else 0
    car = 1 if car_text == "Y" else 0
    property = 1 if property_text == "Y" else 0

    data = pd.DataFrame([{
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": family_members,
        "AMT_INCOME_TOTAL": income,
        "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": employment_days,

        "credit_score": credit_score,
        "active_loans": active_loans,
        "monthly_emi": monthly_emi,
        "credit_card_count": credit_card_count,

        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "OCCUPATION_TYPE": occupation,

        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": car,
        "FLAG_OWN_REALTY": property,

        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 1,

        "missed_payments": missed_payments,
        "default_history": default_history
    }])
    
    print("========== INPUT DATA ==========")
    print(data)

    print("========== MODEL OUTPUT ==========")
    print(model.predict(data))

    result = model.predict(data)[0]

    if result == "Approved":
       prediction = "✅ Credit Card Approved"
       color = "green"
    else:
       prediction = "❌ Credit Card Rejected"
       color = "red"

    return render_template(
        "index.html",

        fullname=fullname,
        gender=gender_text,
        income=income,
        income_type=income_type,
        education=education,
        family_status=family_status,
        housing=housing,
        employment=employment_days,
        car=car_text,
        property=property_text,
        occupation=occupation,
        family_members=family_members,

        mobile=mobile,
        email=email,
        pan=pan,
        aadhaar=aadhaar,
        address=address,
        city=city,
        state=state,
        pincode=pincode,

        prediction=prediction,
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)