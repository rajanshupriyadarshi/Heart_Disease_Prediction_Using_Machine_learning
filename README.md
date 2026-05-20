# Heart Disease Prediction Web Application

This project is now a Flask-based web application with user registration, user login, admin login, and a protected heart disease prediction dashboard.

## Features

- User registration with full name, email, phone number, and password
- User login before prediction access
- Admin login and admin dashboard
- Heart disease prediction using the trained machine learning model
- User prediction history stored in SQLite
- Styled HTML and CSS interface with Flask routes

## Project Structure

heart-disease-prediction/
- app.py
- static/
  - css/
    - style.css
- templates/
  - base.html
  - index.html
  - register.html
  - login.html
  - user_dashboard.html
  - admin_dashboard.html
- data/
  - app.db
  - heart.csv
- models/
  - model.joblib
  - scaler.joblib
  - metrics.json
- src/
  - predict.py
  - storage.py
  - train.py

## Setup

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

## Run the Flask App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Default Admin Login

- Email: `admin@heartcare.local`
- Password: `Admin@123`

## Notes

- New users must register first, then log in to predict disease risk.
- Admins can view all users and prediction history.
- This project is for demo and learning purposes only, not medical advice.
