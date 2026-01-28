# 🚗 Fuel Efficiency Prediction Using Machine Learning

A comprehensive web application built with Django that leverages advanced machine learning algorithms to predict vehicle fuel efficiency based on various vehicle specifications. This system provides both administrative controls for model training and user-friendly prediction interfaces.

![Python Version](https://img.shields.io/badge/python-3.7-blue.svg)
![Django Version](https://img.shields.io/badge/django-2.2.7-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20Gradient%20Boosting-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Machine Learning Models](#-machine-learning-models)
- [API Endpoints](#-api-endpoints)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Admin Features
- **Dataset Management**: Upload and manage vehicle datasets in CSV format
- **Data Preprocessing**: Automated data cleaning, missing value handling, and categorical encoding
- **Model Training**: Build and train multiple ML models (Random Forest & Gradient Boosting)
- **Performance Comparison**: Visual comparison of model accuracies with interactive graphs
- **Model Persistence**: Save and load trained models for future predictions

### User Features
- **User Authentication**: Secure registration and login system
- **Fuel Efficiency Prediction**: Predict fuel efficiency based on vehicle specifications
- **Interactive Interface**: User-friendly forms for entering vehicle data
- **Real-time Results**: Get instant predictions using pre-trained models

## 🛠 Technology Stack

### Backend
- **Framework**: Django 2.2.7
- **Language**: Python 3.7
- **Database**: SQLite3

### Machine Learning
- **Libraries**:
  - scikit-learn 0.24.2 (Random Forest, Gradient Boosting)
  - TensorFlow 2.5.0 (Deep Learning capabilities)
  - pandas 1.1.5 (Data manipulation)
  - joblib 1.0.1 (Model persistence)
  - matplotlib 3.3.4 (Data visualization)

### Frontend
- **HTML5/CSS3**: Responsive templates
- **Bootstrap**: Modern UI components

## 🏗 Project Architecture

```
Fuel Efficiency Prediction/
│
├── fuel_consumption/          # Django project settings
│   └── settings.py           # Configuration
│
├── fuel_app/                 # Main application
│   ├── models.py            # Database models
│   ├── views.py             # Business logic
│   ├── forms.py             # Django forms
│   └── admin.py             # Admin configurations
│
├── Templates/               # HTML templates
│   ├── index.html          # Landing page
│   ├── admin_login.html    # Admin authentication
│   ├── admin_home.html     # Admin dashboard
│   ├── upload_dataset.html # Dataset upload
│   ├── preprocess_dataset.html # Data preprocessing
│   ├── build_model.html    # Model training
│   ├── user_register.html  # User registration
│   ├── user_login.html     # User authentication
│   ├── user_home.html      # User dashboard
│   ├── enter_test_data.html # Prediction interface
│   └── comparison.html     # Model comparison
│
├── models/                  # Trained ML models
│   ├── fuel_efficiency_rf.pkl    # Random Forest model
│   └── fuel_efficiency_gbr.pkl   # Gradient Boosting model
│
├── datasets/               # Dataset storage
│   └── Cars_India_dataset.csv
│
├── static/                 # Static files (CSS, JS, images)
│
├── manage.py              # Django management script
└── requirements.txt       # Python dependencies
```

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/fuel-efficiency-prediction.git
cd fuel-efficiency-prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Database Migration
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 5: Run Development Server
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## 🎯 Usage

### Admin Workflow

1. **Login as Admin**
   - Navigate to `/admin_login`
   - Default credentials: Username: `Admin`, Password: `Admin`

2. **Upload Dataset**
   - Go to "Upload Dataset" section
   - Upload a CSV file containing vehicle data
   - Required columns: Displacement, Seats, Length, Width, Height, Wheelbase, No_of_Cylinders, Fuel Tank Capacity, Boot Space, Fuel, Transmission, Drive, Fuel Efficiency

3. **Preprocess Dataset**
   - Navigate to "Preprocess Dataset"
   - Click "Preprocess" to clean and encode data
   - View preprocessed data preview

4. **Build Model**
   - Go to "Build Model" section
   - Train Random Forest and Gradient Boosting models
   - View performance metrics (R², MAE, RMSE, Accuracy)
   - Compare models using generated graphs

### User Workflow

1. **Register/Login**
   - Create a new account at `/user_register`
   - Login with credentials at `/user_login`

2. **Enter Vehicle Data**
   - Navigate to "Predict Fuel Efficiency"
   - Fill in vehicle specifications:
     - Displacement (cc)
     - Number of Seats
     - Dimensions (Length, Width, Height in mm)
     - Wheelbase (mm)
     - Number of Cylinders
     - Fuel Tank Capacity (liters)
     - Boot Space (liters)
     - Fuel Type (0: Diesel, 1: Petrol, 2: CNG, etc.)
     - Transmission (0: Manual, 1: Automatic)
     - Drive Type (0: FWD, 1: RWD, 2: AWD)

3. **Get Prediction**
   - Click "Predict"
   - View predicted fuel efficiency in km/l

## 🤖 Machine Learning Models

### 1. Random Forest Regressor
- **Parameters**:
  - n_estimators: 300
  - max_depth: 12
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

- **Performance**: Typically achieves 85-95% R² score on test data

### 2. Gradient Boosting Regressor
- **Parameters**:
  - n_estimators: 300
  - learning_rate: 0.05
  - max_depth: 6
  - random_state: 42

- **Performance**: Typically achieves 90-95% R² score on test data

### Feature Engineering
- **Categorical Encoding**: LabelEncoder for Fuel, Transmission, Drive
- **Feature Selection**: 12 key features used for prediction
- **Data Preprocessing**: Missing value removal and normalization

### Model Evaluation Metrics
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Accuracy**: R² × 100 (percentage)

## 🌐 API Endpoints

### Public Routes
- `/` - Home page
- `/user_register/` - User registration
- `/user_login/` - User login

### Admin Routes
- `/admin_login/` - Admin login
- `/admin_home/` - Admin dashboard
- `/upload_dataset/` - Dataset upload
- `/preprocess_dataset/` - Data preprocessing
- `/build_model/` - Model training
- `/admin_logout/` - Admin logout

### User Routes (Authentication Required)
- `/user_home/` - User dashboard
- `/enter_test_data/` - Prediction interface
- `/comparison/` - Model comparison view
- `/user_logout/` - User logout

## 📊 Dataset Format

The application expects CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Displacement | Float | Engine displacement in cc |
| Seats | Integer | Number of seats |
| Length | Float | Vehicle length in mm |
| Width | Float | Vehicle width in mm |
| Height | Float | Vehicle height in mm |
| Wheelbase | Float | Wheelbase in mm |
| No_of_Cylinders | Integer | Number of cylinders |
| Fuel Tank Capacity | Float | Tank capacity in liters |
| Boot Space | Float | Boot space in liters |
| Fuel | String | Fuel type (Petrol/Diesel/CNG) |
| Transmission | String | Transmission type (Manual/Automatic) |
| Drive | String | Drive type (FWD/RWD/AWD) |
| Fuel Efficiency | Float | Target variable (km/l) |

## 🎨 Screenshots

*Add your screenshots here showcasing:*
- Admin dashboard
- Dataset upload interface
- Model training results
- User prediction interface
- Performance comparison graphs

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Add more ML algorithms (XGBoost, Neural Networks)
- [ ] Implement advanced visualization dashboards
- [ ] Add API endpoints for external integration
- [ ] Support for multiple dataset formats (Excel, JSON)
- [ ] Real-time model retraining
- [ ] Export predictions to PDF/CSV
- [ ] Add feature importance analysis
- [ ] Implement A/B testing for models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset source: Cars India Dataset
- Machine Learning libraries: scikit-learn, TensorFlow
- Web framework: Django
- UI inspiration: Bootstrap

## 📞 Support

For support, email your.email@example.com or create an issue in the repository.

---

**⭐ If you found this project helpful, please consider giving it a star!**
