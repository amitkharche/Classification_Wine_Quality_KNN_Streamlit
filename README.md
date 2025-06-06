
# 🍷 Wine Quality Prediction using K-Nearest Neighbors (KNN)

This end-to-end machine learning project predicts **wine quality scores** using the **K-Nearest Neighbors (KNN)** algorithm. It includes:

✅ Exploratory Data Analysis  
✅ Data Preprocessing & Feature Scaling  
✅ GridSearchCV-based Hyperparameter Tuning  
✅ Model Evaluation and Performance Visualization  
✅ SHAP Explainability for Individual Predictions  
✅ Streamlit App with Dashboard  
✅ Dockerfile, GitHub CI/CD, and Cloud Deployment Support

---

## 📸 Demo Snapshots

| 🔮 SHAP-based Prediction | 📊 Performance Dashboard |
|--------------------------|--------------------------|
| ![SHAP GIF](assets/shap_demo.gif) | ![Dashboard GIF](assets/dashboard_demo.gif) |

> ⚠️ Replace above paths with your actual GitHub `raw` file URLs if deploying online.

---

## 🗂️ Project Structure

```
wine-quality-knn-project/
├── data/                    # Dataset CSV
├── models/                  # Saved KNN model and scaler
├── output/                  # Evaluation reports and plots
├── scripts/                 # Modular Python scripts
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
├── notebooks/              # EDA + modeling notebook
├── streamlit_app/          # Streamlit UI app
├── .github/workflows/      # GitHub CI workflow
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Docker Compose config
├── requirements.txt        # Project dependencies
└── README.md               # Project overview and usage
```

---

## 📦 Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model (With GridSearch)
```bash
python scripts/train.py --use_grid_search
```

### 3️⃣ Evaluate the Model
```bash
python scripts/evaluate.py
```

### 4️⃣ Launch Streamlit App
```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 Streamlit App Features

### 🔢 Real-Time Prediction Tab:
- Takes 11 wine features as input
- Displays predicted `quality` score
- Visual SHAP waterfall plot to explain input impact

### 📊 Dashboard Tab:
- Confusion matrix heatmap
- Classification report (accuracy, precision, recall, F1)
- Supports full test-set predictions from saved model

---

## 🧠 SHAP Explainability

- Uses `shap.Explainer` to analyze how each input feature affects model output
- Waterfall plot shows direction and magnitude of each feature's influence
- Helps users interpret **why** the model predicted a certain wine quality

---

## 🐳 Docker Setup

### Build & Run Manually
```bash
docker build -t wine-knn-app .
docker run -p 8501:8501 wine-knn-app
```

### With Docker Compose
```bash
docker-compose up --build
```

---

## 🤖 CI/CD Pipeline (GitHub Actions)

Basic CI is configured to:
- Run model training on every push
- Can be extended to include test suite, linting, deployment

`.github/workflows/ci.yml`

---

## 🧪 Dataset Source

- UCI Wine Quality Dataset  
- [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- Features: acidity, sugar, alcohol, etc.  
- Target: `quality` score (integer 0–10)

---

## 🛠 Future Enhancements

- [ ] Compare KNN with RandomForest, LightGBM
- [ ] Add model drift detection module
- [ ] Deploy to Streamlit Cloud or HuggingFace Spaces
- [ ] Use MLflow for experiment tracking

---

## 🖼 Screenshots (Optional)

Add screenshots or screen recordings to the `assets/` folder:
```
assets/
├── shap_demo.gif
├── dashboard_demo.gif
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and set:
   - Main file: `streamlit_app/app.py`
   - Dependencies: `requirements.txt`

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

Open issues and submit pull requests for improvements, bug fixes, or new features. Contributions are welcome!

---

