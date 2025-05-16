# 💳Credit Card Fraud Detection Platform

## 📌1. Project Overview
This project provides an end-to-end data insight platform for real-time credit card fraud detection, featuring:
- **Real-Time Data Simulation** with timestamped entries and fraud patterns
- **ML Fraud Detection** model trained with synthetic and complex patterns
- **Interactive Streamlit Dashboard** for live visualization and prediction
- **RAG-based Chatbot** using Pinecone + Gemini API for contextual document Q&A
- **Dataset kaggle link** - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 🛠️ Project Structure

```bash
.
├── data-simulation.py       # Real-time transaction data generator
├── model.py                 # ML pipeline: training, evaluation, model saving
├── stream.py                # Streamlit dashboard for real-time prediction
├── chatbot.py               # RAG-based Q&A chatbot using Gemini & Pinecone
├── final.py                 # Final Integration - Merge dashboard and chatbot in a single Streamlit interface
├── fraud_detection_pipeline.joblib  # Trained ML pipeline (generated after training)
├── feature_names.joblib 
├── creditcard.csv           # Dataset csv file    
└── README.md
```


## 📌2.  Setup and execution instructions
---**Prerequisites**---
•	Python 3.8 or higher
•	Pip package manager
---**Installation**---
1.	Install Required libraries include:
            streamlit
            scikit-learn
            pandas
            numpy
            joblib
            matplotlib, seaborn, plotly
            langchain, pymupdf, pinecone-client, google-generativeai

2. Download the Credit Card Fraud dataset:
o	Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
o	Download the CSV file and place it in the data directory

3.	Set up environment variables (for RAG implementation):
# Linux/Mac
export GEMINI_API_KEY=AIzaSyBDbgnmAo705WKzXtGQVykpnxJtXUpUQdk
export PINECONE_API_KEY=pcsk_7PnniW_AZxLcRnGLGvRNdVJL6xqHUWqZD5KwfFyfSq5K1PFb6RD7PPw2HFrjZqTes2F7Ug
export PINECONE_ENV=us-east-1

# Windows
set GEMINI_API_KEY=AIzaSyBDbgnmAo705WKzXtGQVykpnxJtXUpUQdk
set PINECONE_API_KEY=pcsk_7PnniW_AZxLcRnGLGvRNdVJL6xqHUWqZD5KwfFyfSq5K1PFb6RD7PPw2HFrjZqTes2F7Ug
set PINECONE_ENV=us-east-1

---**Running the Application**---
1.	## Train ML(fraud detection) model:
        
        python model.py
        
    This will:
        ○ Generate synthetic fraud/legit data
        ○ Train a RandomForestClassifier
        ○ Save the pipeline to fraud_detection_pipeline.joblib

2.	## Launch the Streamlit dashboard:

        python -m streamlit run final.py

    This will:
        ○ Streams live transaction data
        ○ Predicts fraud in real-time
        ○ Displays plots and transaction logs
        ○ PDF-based chatbot
        ○ Answer questions about fraud detection, Python, ML, and more
        ○ Uses Pinecone vector search + Gemini LLM for answers


3.	## Access the dashboard in your web browser:
        http://localhost:8501


## 📌3. Model and tool explanation

**Model Type:** RandomForestClassifier
**Data Simulation:** Synthetic patterns with fraud heuristics
**Preprocessing:** SMOTE oversampling + StandardScaler
**Evaluation Metrics:** Accuracy, ROC-AUC, Confusion Matrix
**Fraud Patterns Modeled:** Late-night transactions, High/low amount anomalies, Sparse transaction activity

## Chatbot Tech Stack
**Embedding Model:** Gemini Embedding (Google Generative AI)
**Vector Store:** Pinecone
**PDF Loader:** PyMuPDF
**RAG Chain:** LangChain's ConversationalRetrievalChain
**Frontend:** Streamlit Chat UI

## 📌4. Capture and submit screenshots of:
○ **Dashboard**                    
![alt text](<screenshots/Chatbot dashboard.png>)                    # Chatbot Dashboard before selecting option
![alt text](<screenshots/Real-time-prediction dashboard.png>)       # Real-time prediction dashboard
![alt text](screenshots/dashboard-with-options.png)                 # Dashboard with 2 options


○ **Real-time prediction** 
![alt text](<screenshots/Dataset information.png>)                  
# Dataset information - Dataset Shape, Number of Fraudulent Transactions, Sample Data
![alt text](screenshots/simulation.png)
# Simulation with plot with recent transactions, Transaction Volume, Fraud Probability Distribution
![alt text](<screenshots/data with legit-fraud.png>)
# Data with Legit and fraud transaction
![alt text](<screenshots/Filter category.png>)
# Filter category - filter with simulation speed, average time between transactions, fraud probability


○ **Chatbot with relevant answers** 
![alt text](<screenshots/chatbot answer -1.png>)
![alt text](<screenshots/chatbot answer- 2.png>)
![alt text](<screenshots/chatbot answer -3.png>)






