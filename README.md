# 🏠 House Price Prediction - AI/ML Internship Task 6

## ✅ Task Objective
Predict house prices using property features such as area, bedrooms, bathrooms, floors, and location.  
The goal is to explore the dataset, preprocess features, train a regression model, and evaluate performance using metrics like MAE and RMSE.

## 📂 Dataset Used
- **Dataset Name:** House Price Prediction Dataset  
- **Source:** Uploaded CSV (`/dataset/House Price Prediction Dataset.csv`)  
- **Features:**  
  - Area (numeric)  
  - Bedrooms (numeric)  
  - Bathrooms (numeric)  
  - Floors (numeric)  
  - Location (categorical)  
- **Target:** Price (numeric)

## 📂 GitHub Repo Structure

```
HousePricePrediction/
│
├── README.md
├── house_price_prediction.ipynb       # Jupyter Notebook
├── house_price_prediction.py          # Python Script version
└── dataset/                           # Folder to place your CSV
     └── House Price Prediction Dataset.csv
```

## 🧪 Models Applied
- Gradient Boosting Regressor (Scikit-Learn)  
- Optional: Linear Regression for baseline comparison

## 📊 Key Steps
1. Load dataset and inspect columns  
2. Select relevant features and target  
3. Encode categorical variables (`Location`) using OneHotEncoder  
4. Train-test split (80%-20%)  
5. Train Gradient Boosting Regressor  
6. Make predictions on test set  
7. Evaluate performance using **MAE** and **RMSE**  
8. Visualize predictions vs actual prices and error distribution

## 📈 Key Results
<img width="444" height="617" alt="image" src="https://github.com/user-attachments/assets/c3c8fa05-c36f-407d-b7ca-2f7ffb3625eb" />
  
- Visualizations show predicted prices closely follow actual prices  
- Model effectively captures trends in property prices based on features  

## 📌 Insights
- `Location` and `Area` are strong predictors  
- Gradient Boosting Regressor performs better than simple linear regression  
- Prediction errors are mostly small and centered around zero

## ▶️ How to Run
1. Place the dataset in `/dataset/` folder  
2. Open `house_price_prediction.ipynb` in Jupyter Notebook or Google Colab and run all cells  
3. Alternatively, run the script version:
```bash
python house_price_prediction.py
```

## 🤝 Contributing

Enhancements welcome:

Add more features like YearBuilt, Garage, Condition

Try other regression models like XGBoost or RandomForest

Hyperparameter tuning for improved performance

## 📜 License

MIT License
