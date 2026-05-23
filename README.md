# PAU Student Performance Prediction App
**Pan-Atlantic University · Lagos, Nigeria**

A Streamlit web application for predicting student academic risk using a trained Random Forest classifier.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure
```
pau_app/
├── app.py                  ← Main Streamlit application
├── student_data.csv        ← Training dataset (150 students)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── models/                 ← Auto-generated on first run
│   └── random_forest_model.pkl
└── .streamlit/
    └── config.toml         ← Theme configuration
```

---

## 🎯 Features
| Feature | Description |
|---|---|
| **Dashboard** | KPI cards, risk distribution charts, level-wise analysis |
| **Single Prediction** | Enter student details in the sidebar → instant risk result |
| **Batch Prediction** | Upload a CSV of students → get predictions for all |
| **Model Performance** | Confusion matrix, ROC curve, classification report |
| **EDA** | Feature distributions, correlation heatmap, box plots |
| **Student Data** | Searchable, filterable table with colour-coded risk labels |
| **Export** | Download batch prediction results as CSV |

---

## 📊 Dataset Columns
| Column | Type | Description |
|---|---|---|
| student_id | int | Unique student identifier |
| name | str | Student full name |
| level | int | Academic level (100–400) |
| department | str | Department name |
| attendance_percentage | int | % of classes attended |
| ca_score | int | Continuous Assessment score |
| exam_score | int | Examination score |
| previous_gpa | float | GPA from previous semester |
| at_risk | int | Target: 1 = At-Risk, 0 = Not At-Risk |

---

## 🤖 Model Details
- **Algorithm:** Random Forest Classifier (100 trees)
- **Features:** level, attendance_percentage, ca_score, exam_score, previous_gpa
- **Train/Test Split:** 80% / 20%
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

*© 2026 Pan-Atlantic University · Unlocking Potential · Building Responsible Leaders*
