# === Imports ===
import os
import time
import tempfile
import requests
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from collections import deque
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LangChain & Pinecone
import pymupdf
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import Any, List

# === RAG Chatbot Components ===

class CustomPineconeRetriever(BaseRetriever):
    pinecone_index: Any = Field(description="Pinecone index")
    embedding_function: Any = Field(description="Embedding function")
    namespace: str = Field(description="Namespace")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            namespace=self.namespace,
            include_metadata=True
        )
        docs = []
        for match in results.matches:
            metadata = match.metadata
            text = metadata.pop('text', '')
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

class RAGChatbot:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
        self.pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "")
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.pdf_urls = [
            "https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf",
            "https://pandas.pydata.org/docs/pandas.pdf",
            "https://buildmedia.readthedocs.org/media/pdf/streamlit/stable/streamlit.pdf",
            "https://faculty.washington.edu/mdecock/papers/pmlr-v72-lopez-rojas18a.pdf",
            "https://www.aaai.org/Papers/KDD/1998/KDD98-011.pdf"
        ]

        self.vector_store = None
        self.custom_retriever = None
        self.conversation_chain = None
        self.index_name = "credit-card-fraud-chatbot"
        self.use_custom_retriever = False

    def download_pdf(self, url, output_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def extract_text_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()

    def setup_pinecone(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def process_pdfs(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            all_documents = []
            for i, url in enumerate(self.pdf_urls):
                pdf_path = os.path.join(temp_dir, f"document_{i}.pdf")
                if not self.download_pdf(url, pdf_path):
                    continue
                documents = self.extract_text_from_pdf(pdf_path)
                chunks = text_splitter.split_documents(documents)
                all_documents.extend(chunks)

            self.setup_pinecone()
            texts = [doc.page_content for doc in all_documents]
            metadatas = [doc.metadata for doc in all_documents]
            embeddings_list = embeddings.embed_documents(texts)
            index = self.pc.Index(self.index_name)
            vectors = [{
                'id': f'doc_{i}',
                'values': embedding,
                'metadata': {**metadata, 'text': text}
            } for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings_list))]

            for i in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[i:i+100], namespace="credit-card-fraud-docs")

            self.custom_retriever = CustomPineconeRetriever(
                pinecone_index=index,
                embedding_function=embeddings,
                namespace="credit-card-fraud-docs"
            )
            self.use_custom_retriever = True
            return self.custom_retriever

    def initialize_chain(self):
        if not self.custom_retriever:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name in existing_indexes:
                index = self.pc.Index(self.index_name)
                self.custom_retriever = CustomPineconeRetriever(
                    pinecone_index=index,
                    embedding_function=embeddings,
                    namespace="credit-card-fraud-docs"
                )
                self.use_custom_retriever = True
            else:
                self.process_pdfs()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-1.5-pro",
                google_api_key=self.gemini_api_key
            ),
            retriever=self.custom_retriever,
            memory=memory
        )
        return self.conversation_chain

    def ask(self, question):
        if not self.conversation_chain:
            self.initialize_chain()
        if not self.conversation_chain:
            return "Failed to initialize the chatbot."
        try:
            response = self.conversation_chain({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Error: {str(e)}"

# === Streamlit Dashboard Functions ===

@st.cache_data
def load_data():
    try:
        return pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        st.warning("Using a synthetic dataset for demo.")
        np.random.seed(42)
        n_samples = 1000
        features = np.random.randn(n_samples, 28)
        time_values = np.sort(np.random.uniform(0, 172800, n_samples))
        amounts = np.random.exponential(scale=100, size=n_samples)
        classes = np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        data = np.column_stack([time_values, features, amounts, classes])
        df = pd.DataFrame(data, columns=columns)
        df['Class'] = df['Class'].astype(int)
        return df

@st.cache_resource
def train_model(df):
    features = [col for col in df.columns if col.startswith('V') or col == 'Amount']
    X = df[features]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, accuracy_score(y_test, y_pred), X_test, y_test, features

def generate_transaction(df, features, timestamp, speed_factor=1.0):
    sample = df.sample(1)[features].iloc[0].copy()
    for feature in features:
        sample[feature] += np.random.normal(0, 0.1) if feature != 'Amount' else np.random.uniform(-0.2, 0.2) * sample[feature]
    return pd.Series({'Timestamp': timestamp, **sample})

# === Streamlit Main App ===

def chatbot_ui():
    st.title("💬 Credit Card Fraud Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def dashboard_ui():
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
        
# === App Selection ===

def main():
    os.environ["GEMINI_API_KEY"] = "AIzaSyBDbgnmAo705WKzXtGQVykpnxJtXUpUQdk"  
    os.environ["PINECONE_API_KEY"] = "pcsk_7PnniW_AZxLcRnGLGvRNdVJL6xqHUWqZD5KwfFyfSq5K1PFb6RD7PPw2HFrjZqTes2F7Ug" 
    os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

    app_mode = st.sidebar.selectbox("Choose App Mode", ["Chatbot", "Fraud Dashboard"])
    if app_mode == "Chatbot":
        chatbot_ui()
    else:
        dashboard_ui()

if __name__ == "__main__":
    main()
