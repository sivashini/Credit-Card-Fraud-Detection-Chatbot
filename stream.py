import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('creditcard.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please download from Kaggle and save as 'creditcard.csv'")
        # For demo purposes, create a small sample dataset
        st.warning("Creating a sample dataset for demonstration purposes")
        
        # Create a synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Features V1-V28 (normalized PCA components)
        features = np.random.randn(n_samples, 28)
        
        # Time (seconds elapsed between transactions)
        time_values = np.sort(np.random.uniform(0, 172800, n_samples))  # 48 hours in seconds
        
        # Amount (transaction amount)
        amounts = np.random.exponential(scale=100, size=n_samples)
        
        # Class (fraud or not, with about 0.2% fraud rate)
        classes = np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])
        
        # Create DataFrame
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        data = np.column_stack([time_values, features, amounts, classes])
        df = pd.DataFrame(data, columns=columns)
        
        # Ensure correct data types
        df['Class'] = df['Class'].astype(int)
        
        return df

# Function to train model
@st.cache_resource
def train_model(df):
    # Feature selection (use all V features and Amount)
    features = [col for col in df.columns if col.startswith('V') or col == 'Amount']
    X = df[features]
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return model and performance metrics
    return model, accuracy, X_test, y_test, features

# Function to generate simulated transactions
def generate_transaction(df, features, timestamp, speed_factor=1.0):
    # Get a random sample from the real data
    sample_idx = np.random.randint(0, len(df))
    new_transaction = df.iloc[sample_idx][features].copy()
    
    # Add some randomness
    for feature in features:
        if feature == 'Amount':
            # For amount, add noise proportional to the value
            noise_factor = 0.2  # 20% noise
            noise = np.random.uniform(-noise_factor, noise_factor) * new_transaction[feature]
            new_transaction[feature] += noise
        else:
            # For other features, add small random noise
            noise = np.random.normal(0, 0.1)
            new_transaction[feature] += noise
    
    # Create a Series with the new transaction
    transaction = pd.Series(new_transaction)
    
    # Add timestamp
    transaction_with_time = pd.Series({'Timestamp': timestamp, **transaction})
    
    return transaction_with_time

# Main app
def main():
    st.title("💳 Credit Card Fraud Detection Dashboard")
    
    # Load data
    with st.spinner("Loading and preparing data..."):
        df = load_data()
    
    # Show dataset info in an expander
    with st.expander("Dataset Information"):
        st.write(f"Dataset Shape: {df.shape}")
        st.write(f"Number of Fraudulent Transactions: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
        st.write("Sample Data:")
        st.dataframe(df.sample(5))
    
    # Train model
    with st.spinner("Training fraud detection model..."):
        model, accuracy, X_test, y_test, features = train_model(df)
        st.sidebar.success(f"Model Accuracy: {accuracy:.4f}")
    
    # Sidebar controls
    st.sidebar.header("Simulation Controls")
    simulation_speed = st.sidebar.slider("Simulation Speed", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    transaction_interval = st.sidebar.slider("Average Time Between Transactions (sec)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    running = st.sidebar.checkbox("Start Simulation", value=False)
    
    # Define container for real-time transactions
    transaction_container = st.container()
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    # Initialize data storage for streaming
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=['Timestamp'] + features + ['Predicted', 'Probability'])
        st.session_state.start_time = time.time()
        st.session_state.transaction_times = []
        st.session_state.fraud_counts = deque(maxlen=100)
        st.session_state.legit_counts = deque(maxlen=100)
        st.session_state.chart_timestamps = deque(maxlen=100)
    
    # Real-time charts
    with col1:
        st.subheader("Transaction Volume")
        volume_chart = st.empty()
    
    with col2:
        st.subheader("Fraud Probability Distribution")
        fraud_dist_chart = st.empty()
    
    # Create container for fraud alerts
    alert_container = st.container()
    with alert_container:
        st.subheader("⚠️ Fraud Alerts")
        alert_placeholder = st.empty()
    
    # Real-time transaction list
    with transaction_container:
        st.subheader("Recent Transactions")
        transaction_table = st.empty()
    
    # Main simulation loop
    if running:
        # Clear previous alerts
        alert_placeholder.empty()
        fraud_alerts = []
        
        # Update timestamp
        current_time = time.time() - st.session_state.start_time
        
        # Simulate a new transaction
        wait_time = np.random.exponential(scale=transaction_interval/simulation_speed)
        time.sleep(wait_time)
        
        transaction = generate_transaction(df, features, current_time, speed_factor=simulation_speed)
        
        # Make prediction
        feature_values = transaction[features].values.reshape(1, -1)
        fraud_probability = model.predict_proba(feature_values)[0][1]
        prediction = 1 if fraud_probability > 0.7 else 0
        
        # Add prediction to transaction
        transaction_with_pred = pd.Series({
            **transaction.to_dict(),
            'Predicted': prediction,
            'Probability': fraud_probability
        })
        
        # Add to dataframe
        new_row_df = pd.DataFrame([transaction_with_pred])
        st.session_state.transactions = pd.concat([new_row_df, st.session_state.transactions]).reset_index(drop=True)
        
        # Keep only the most recent 100 transactions
        if len(st.session_state.transactions) > 100:
            st.session_state.transactions = st.session_state.transactions.iloc[:100]
        
        # Update session state for charts
        st.session_state.transaction_times.append(current_time)
        if prediction == 1:
            st.session_state.fraud_counts.append(1)
            st.session_state.legit_counts.append(0)
            # Add to alerts
            fraud_alerts.append({
                'time': current_time,
                'amount': transaction['Amount'],
                'probability': fraud_probability
            })
        else:
            st.session_state.fraud_counts.append(0)
            st.session_state.legit_counts.append(1)
        
        st.session_state.chart_timestamps.append(current_time)
        
        # Update transaction table
        display_df = st.session_state.transactions[['Timestamp', 'Amount', 'Predicted', 'Probability']].copy()
        display_df['Timestamp'] = display_df['Timestamp'].round(2)
        display_df['Amount'] = display_df['Amount'].round(2)
        display_df['Probability'] = (display_df['Probability'] * 100).round(2).astype(str) + '%'
        display_df['Type'] = display_df['Predicted'].map({0: 'Legitimate', 1: 'Fraudulent'})
        display_df = display_df.drop(columns=['Predicted'])
        
        # Apply styling
        def highlight_fraud(row):
            if row['Type'] == 'Fraudulent':
                return ['background-color: rgba(255, 0, 0, 0.3)'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_fraud, axis=1)
        transaction_table.dataframe(styled_df)
        
        # Create transaction volume chart
        if len(st.session_state.chart_timestamps) > 0:
            timestamps = list(st.session_state.chart_timestamps)
            fraud_counts = list(st.session_state.fraud_counts)
            legit_counts = list(st.session_state.legit_counts)
            
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Scatter(
                x=timestamps, 
                y=legit_counts,
                name='Legitimate',
                fill='tozeroy',
                line=dict(color='green')
            ))
            volume_fig.add_trace(go.Scatter(
                x=timestamps, 
                y=fraud_counts,
                name='Fraudulent',
                fill='tozeroy',
                line=dict(color='red')
            ))
            volume_fig.update_layout(
                xaxis_title='Time (s)',
                yaxis_title='Count',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            volume_chart.plotly_chart(volume_fig, use_container_width=True)
        
        # Create fraud probability distribution
        if len(st.session_state.transactions) > 0:
            fraud_fig = px.histogram(
                st.session_state.transactions, 
                x='Probability',
                nbins=20,
                color_discrete_sequence=['blue'],
                opacity=0.7
            )
            fraud_fig.add_vline(x=0.7, line_dash="dash", line_color="red")
            fraud_fig.update_layout(
                xaxis_title='Fraud Probability',
                yaxis_title='Count',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            fraud_dist_chart.plotly_chart(fraud_fig, use_container_width=True)
        
        # Show fraud alerts
        if fraud_alerts:
            alert_text = ""
            for alert in fraud_alerts:
                alert_text += f"🚨 Potential fraud detected at {alert['time']:.2f}s - Amount: ${alert['amount']:.2f} (Probability: {alert['probability']:.2%})\n\n"
            alert_placeholder.markdown(alert_text)
        
        # Auto-rerun to simulate continuous streaming
        st.experimental_rerun()
        #st.rerun()


if __name__ == "__main__":
    main()