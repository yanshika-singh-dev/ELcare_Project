# 🏥 ElCare — AI-Powered Medical Disease Predictor

> An intelligent early detection system for Parkinson's Disease, Heart Disease, and Diabetes using Machine Learning.


---

## 📌 About

**ElCare** is a web-based medical prediction application that uses trained Machine Learning models to predict the likelihood of three major diseases based on clinical input data. It features user authentication, patient profile management, and full prediction history tracking.

---

## ✨ Features

- 🧠 **Parkinson's Disease Prediction** — using 22 voice measurement features (SVM model)
- ❤️ **Heart Disease Prediction** — using 13 medical parameters (Logistic Regression)
- 💉 **Diabetes Prediction** — using 8 health indicators (SVM model)
- 🔐 **User Authentication** — Register, Login, Logout with hashed passwords
- 👤 **Patient Profile** — Store personal and medical info with live BMI calculator
- 📋 **Prediction History** — View, filter and delete past predictions
- 💾 **SQLite Database** — Stores users, profiles and prediction records

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Models | Scikit-learn (SVM, Logistic Regression) |
| Database | SQLite via Flask-SQLAlchemy |
| Auth | Flask-Login, Werkzeug password hashing |
| Frontend | HTML, CSS, JavaScript |
| Environment | python-dotenv |

---

## 📁 Project Structure

```
ElCare/
├── app.py                  # Main Flask application
├── models.py               # Database models (User, PatientProfile, PredictionHistory)
├── auth.py                 # Authentication routes (login, register, logout)
├── profile_routes.py       # Profile and history API routes
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not pushed to GitHub)
├── data/
│   ├── parkinsons.data
│   ├── heart_disease_data.csv
│   └── diabetes (2).csv
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── history.html
│   ├── parkinsons.html
│   ├── heart.html
│   └── diabetes.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yanshikasingh055-spec/ELcare_Project
cd ElCare
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url 
```

### 5. Run the application
```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## 🗄️ Database

The app uses **SQLite** by default — no setup required. The `elcare.db` file is auto-created on first run with these tables:

| Table | Description |
|-------|-------------|
| `users` | Stores user accounts |
| `patient_profiles` | Stores medical profile per user |
| `prediction_history` | Stores all past predictions |

To switch to MySQL, update `.env`:
```
DATABASE_URL=mysql+pymysql://root:password@localhost/elcare_db
```

---

## 🤖 ML Models

| Disease | Algorithm | Features | Accuracy |
|---------|-----------|----------|----------|
| Parkinson's | SVM (Linear Kernel) | 22 | ~88% |
| Heart Disease | Logistic Regression | 13 | ~85% |
| Diabetes | SVM (Linear Kernel) | 8 | ~78% |

Models are trained on startup using the datasets in the `/data` folder.

---

## 📸 Screenshots

### Home Page
![Home](screenshots/home.png)

### Sign Up Page 
![SignUp](screenshots/Signup.png)

### Login Page
![Login](screenshots/login.png)

### Prediction Page
![Prediction](screenshots/prediction.png)

---

## ⚠️ Disclaimer

This application is built for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 👩‍💻 Author

**Yanshika Singh**
- GitHub: [@yanshikasingh055-spec](https://github.com/yanshikasingh055-spec)
- Email: yanshikasingh055@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

