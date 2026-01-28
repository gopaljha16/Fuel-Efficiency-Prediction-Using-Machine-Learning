from django.shortcuts import render, redirect
import os
import pandas as pd
from .models import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import joblib  # for saving model
from .forms import DatasetForm
# Add these imports for user auth
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')




# Create your views here.
def index(request):
    return render(request,'index.html')



# Admin login
def admin_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        if username == "Admin" and password == "Admin":
            request.session['admin_logged_in'] = True
            return redirect('admin_home')
        else:
            return render(request, 'admin_login.html', {'error': 'Invalid Username or Password'})
    return render(request, 'admin_login.html')


#admin home
def admin_home(request):
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')
    return render(request, 'admin_home.html')


#admin logout
def admin_logout(request):
    request.session.flush()
    return redirect('index')

# Path to save trained model
model_path = "datasets/fuel_model.pkl"


def upload_dataset(request):
    message = ""
    form = DatasetForm()
    latest_upload = None

    if request.method == "POST":
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            message = "Dataset uploaded successfully!"
            form = DatasetForm()  # clear the form

    latest_upload = Dataset.objects.last()  # ✅ only keep the latest dataset

    return render(
        request,
        "upload_dataset.html",
        {"form": form, "message": message, "latest_upload": latest_upload},
    )


def preprocess_dataset(request):
    dataset = Dataset.objects.last()  # ✅ only the latest one

    preprocessed_data = None
    message = None

    if request.method == "POST" and dataset:
        file_path = dataset.file.path

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # 1️⃣ Drop missing values
            df = df.dropna()

            # 2️⃣ Encode categorical columns
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype("category").cat.codes

            # 3️⃣ Save preprocessed dataset
            preprocessed_file_path = "datasets/preprocessed_dataset.csv"
            os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)

            try:
                df.to_csv(preprocessed_file_path, index=False)
                message = "Dataset preprocessed successfully!"
            except PermissionError:
                message = ("Permission denied: cannot write to 'datasets/preprocessed_dataset.csv'. "
                           "Please close the file if open and check folder permissions.")

            # Show first 10 rows
            preprocessed_data = df.head(10).to_html(
                classes="table table-bordered table-striped"
            )
        else:
            message = "Dataset file not found."

    return render(
        request,
        "preprocess_dataset.html",
        {"dataset": dataset, "preprocessed_data": preprocessed_data, "message": message},
    )

def build_model(request):
    dataset = Dataset.objects.last()
    message = ""
    results = {}

    if dataset:
        file_path = dataset.file.path
        df = pd.read_csv(file_path)

        # Clean data
        df = df.dropna(subset=["Fuel Efficiency"])
        df = df.dropna()

        # Encode categorical columns
        categorical_cols = ["Fuel", "Transmission", "Drive"]
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Features & target
        feature_cols = [
            "Displacement", "Seats", "Length", "Width", "Height",
            "Wheelbase", "No_of_Cylinders", "Fuel Tank Capacity",
            "Boot Space", "Fuel", "Transmission", "Drive"
        ]
        X = df[feature_cols]
        y = df["Fuel Efficiency"]

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ------------------- RANDOM FOREST -------------------
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        results["Random Forest"] = {
            "r2": r2_score(y_test, y_pred_rf),
            "mae": mean_absolute_error(y_test, y_pred_rf),
            "rmse": mean_squared_error(y_test, y_pred_rf, squared=False),
        }
        results["Random Forest"]["accuracy"] = results["Random Forest"]["r2"] * 100

        # Save Random Forest model
        rf_model_path = "models/fuel_efficiency_rf.pkl"
        os.makedirs(os.path.dirname(rf_model_path), exist_ok=True)
        joblib.dump(rf_model, rf_model_path)

        # ------------------- GRADIENT BOOSTING -------------------
        gbr_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        gbr_model.fit(X_train, y_train)
        y_pred_gbr = gbr_model.predict(X_test)

        results["Gradient Boosting"] = {
            "r2": r2_score(y_test, y_pred_gbr),
            "mae": mean_absolute_error(y_test, y_pred_gbr),
            "rmse": mean_squared_error(y_test, y_pred_gbr, squared=False),
        }
        results["Gradient Boosting"]["accuracy"] = results["Gradient Boosting"]["r2"] * 100
        # Save Random Forest model
        gbr_model_path = "models/fuel_efficiency_gbr.pkl"
        os.makedirs(os.path.dirname(gbr_model_path), exist_ok=True)
        joblib.dump(gbr_model, gbr_model_path)


        # ------------------- PLOT GRAPH -------------------
        metrics = ["r2", "mae", "rmse", "accuracy"]
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(metrics))
        width = 0.35

        rf_values = [results["Random Forest"][m] for m in metrics]
        gbr_values = [results["Gradient Boosting"][m] for m in metrics]

        ax.bar([i - width / 2 for i in x], rf_values, width, label="Random Forest")
        ax.bar([i + width / 2 for i in x], gbr_values, width, label="Gradient Boosting")

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylabel("Score")
        ax.set_title("Random Forest vs Gradient Boosting Performance")
        ax.legend()

        # Save graph in static folder
        graph_path = os.path.join("static", "model_comparison.png")
        plt.savefig(graph_path)
        plt.close()

        message = "Random Forest and Gradient Boosting models built successfully!"

    return render(request, "build_model.html", {
        "message": message,
        "results": results,
        "graph": "model_comparison.png"
    })


# User Registration
def user_register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        # Validations
        if not username or not email or not password or not confirm_password:
            messages.error(request, "All fields are required!")
        elif password != confirm_password:
            messages.error(request, "Passwords do not match!")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
        else:
            # Save user to database
            User.objects.create_user(username=username, email=email, password=password)
            messages.success(request, " Registration successful! Please login.")
            return redirect("user_login")  # Redirect to login page

    return render(request, "user_register.html")

"""
# User Login
def user_login(request):
    message = ""
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if not username or not password:
            message = "Please enter both username and password!"
        else:
            try:
                user = User.objects.get(username=username, password=password)
                # Save session
                request.session['user_logged_in'] = True
                request.session['user_id'] = user.id
                request.session['username'] = user.username
                return redirect('user_home')  # Redirect to user home page
            except User.DoesNotExist:
                message = "Invalid username or password!"

    return render(request, "user_login.html", {"message": message})
"""

def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)  # Log in the user
            messages.success(request, f"Welcome, {username}!")
            return redirect('user_home')  # Redirect to home page
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "user_login.html")

def user_logout(request):
    request.session.flush()  # Clear session
    return redirect('user_login')


def comparison(request):
    return render(request,'comparison.html')



@login_required(login_url='user_login')  # Redirect to login if not authenticated
def user_home(request):
    return render(request, "user_home.html", {"username": request.user.username})


def enter_test_data(request):
    result = None
    if request.method == "POST":
        # Load the Gradient Boosting model (Python 3.7 compatible)
        gbr_model_path = "models/fuel_efficiency_gbr.pkl"
        if os.path.exists(gbr_model_path):
            model = joblib.load(gbr_model_path)

            # Collect user input
            displacement = float(request.POST.get("displacement"))
            seats = int(float(request.POST.get("seats")))
            length = float(request.POST.get("length"))
            width = float(request.POST.get("width"))
            height = float(request.POST.get("height"))
            wheelbase = float(request.POST.get("wheelbase"))
            cylinders = int(float(request.POST.get("cylinders")))
            fuel_tank = float(request.POST.get("fuel_tank"))
            boot_space = float(request.POST.get("boot_space"))

            # Dropdown values (already encoded correctly)
            fuel = int(request.POST.get("fuel"))
            transmission = int(request.POST.get("transmission"))
            drive = int(request.POST.get("drive"))

            features = [[displacement, seats, length, width, height,
                          wheelbase, cylinders, fuel_tank, boot_space,
                          fuel, transmission, drive]]

            result = model.predict(features)[0]

    return render(request, "enter_test_data.html", {"result": result})
