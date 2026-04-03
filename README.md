📡 Telecom Customer Churn Predictor

📌 Project Overview
A Machine Learning web application that predicts whether a 
telecom customer will leave the service or stay, based on 
their usage patterns, services and billing details.

🎯 Problem Statement
Telecom companies like Airtel, Jio lose customers every month 
without knowing who is likely to leave. This model predicts 
customer churn so businesses can take action before losing them.

🚀 Live Demo
https://telecom-customer-churn-predictor.streamlit.app/

🛠️ Tech Stack
- Python 3.12
- XGBoost Classifier
- SMOTE (Imbalanced Learn)
- Scikit-learn
- Streamlit
- Pandas & Numpy
- Pickle

📊 Dataset
- Source: IBM Telco Customer Churn Dataset
- 7043 customer records
- 21 features
- Target: Will customer churn? (Yes=1 / No=0)

📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Precision | 0.56 |
| Recall | 0.73 |
| F1 Score | 0.63 |

⚙️ ML Pipeline
1. Real Dataset Loading (IBM Telco)
2. Data Cleaning (Missing values, Drop useless columns)
3. One Hot Encoding (15 categorical columns)
4. SMOTE to handle Imbalanced Data (4138 each class)
5. XGBoost Classifier Training
6. Model Evaluation (Precision, Recall, F1 Score)
7. Model Saved using Pickle
8. Deployed using Streamlit

🔍 Key Features Used
- Customer Demographics (Gender, Age, Partner, Dependents)
- Service Details (Internet, Phone, Streaming, Security)
- Contract Details (Monthly, One year, Two year)
- Billing Details (Monthly Charges, Total Charges, Payment Method)

🖥️ How to Run Locally

        1. Clone the repository
        git clone https://github.com/yourusername/telecom-churn-predictor.git

        2. Install dependencies
        pip install -r requirements.txt

        3. Train the model first
        python customer_prediction.py

        4. Run the app
        streamlit run app.py

📁 Project Structure
telecom_churn/
│
├── customer_prediction.py  ← ML model training code
├── app.py                  ← Streamlit web app
├── model.pkl               ← Saved trained model
├── columns.json            ← Saved column names
├── requirements.txt        ← Dependencies
└── README.md               ← Project documentation

🧠 Key Learnings
- Handling imbalanced datasets using SMOTE
- One Hot Encoding for categorical features
- XGBoost vs Random Forest comparison
- Precision Recall tradeoff in business context
- Recall is most important metric for churn prediction
- End to end ML project deployment using Streamlit

📊 Model Comparison

| Model | Recall | F1 Score |
|-------|--------|----------|
| Random Forest | 0.46 | 0.53 |
| Random Forest + SMOTE | 0.51 | 0.55 |
| XGBoost + SMOTE | 0.73 | 0.63 |

💡 Business Impact
- Identifies 73% of customers likely to leave
- Enables proactive retention strategies
- Suggested actions for each prediction
- Saves customer acquisition costs

👨‍💻 Author

-P M Gowtham
-LinkedIn: https://www.linkedin.com/in/p-m-gowtham-022756355/
-GitHub: https://github.com/Gowtham24MIS
-Email: gowtham.developer07@gmail.com

