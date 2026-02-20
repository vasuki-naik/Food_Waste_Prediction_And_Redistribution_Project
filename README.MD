# Global Food Waste Analysis & Redistribution Potential

End-to-end data analytics and machine learning capstone project that analyzes global food waste trends, forecasts future waste, and estimates the potential impact of food redistribution using an interactive Streamlit dashboard.

---

## Project Overview

Food waste is a major global challenge with significant social, economic, and environmental consequences. This project uses data analytics, machine learning, and impact modelling to explore global food waste patterns and estimate how much food could be recovered to feed people in need.

The project combines a full data pipeline, forecasting models, clustering, and a production-style dashboard.

---

## Key Features

• Data cleaning and feature engineering pipeline  
• Machine learning forecasting of food waste until 2030  
• Country segmentation using clustering (KMeans)  
• Impact modelling (meals saved, people fed, CO₂ reduction)  
• Interactive Streamlit dashboard for exploration and simulation  

---

## Tech Stack

Python  
Pandas & NumPy  
Scikit-Learn  
XGBoost  
Streamlit  
Plotly  

---

## Project Structure

Global-Food-Waste-Analysis/
│
├── app.py                         # Streamlit dashboard  
├── analysis_pipeline.py          # Data pipeline & ML models  
├── requirements.txt              # Project dependencies  
│
├── data/  
│   ├── processed_waste_data_final.csv  
│   └── global_kpi_summary.csv  
│
└── notebooks/  
    └── EDA.ipynb (optional)

---

## Methodology

### 1. Data Pipeline
• Load and clean dataset  
• Handle missing values and duplicates  
• Generate synthetic dataset if source data is unavailable  

### 2. Feature Engineering
Derived metrics were created to estimate real-world impact:

• Recoverable food waste  
• Meals generated from recovered food  
• Number of people fed per year  
• CO₂ emissions saved  

### 3. Machine Learning Forecasting
An XGBoost regression model is trained on historical data (2018–2024) to forecast food waste from 2025–2030 using recursive forecasting and country-specific growth trends.

### 4. Country Segmentation
KMeans clustering groups countries into segments:

• High Waste Economies  
• Growing Waste Countries  
• Efficient Low Waste Countries  

### 5. Impact Modelling
The project estimates the potential global impact of food redistribution:

• Food recovered  
• Meals generated  
• People fed  
• Environmental benefits  

---

## Running the Project

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the data pipeline
python analysis_pipeline.py

This generates the processed datasets used by the dashboard.

### 3. Launch the dashboard
streamlit run app.py

The dashboard will open in your browser.

---

## Dashboard Features

• Global KPI overview  
• Country waste trend & forecast  
• Country comparison and segmentation  
• Food redistribution impact simulator  

---

## Contributors

• Your Name — Data Pipeline & Machine Learning  
• Teammate 1 — Dashboard Development  
• Teammate 2 — EDA & Documentation  

---

## Future Improvements

• Add real global datasets instead of synthetic backup data  
• Add model comparison and confidence intervals  
• Deploy dashboard online  

---

## License

MIT License