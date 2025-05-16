import os
import time
import pymupdf
import requests
import tempfile
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
import streamlit as st


# Create a custom retriever class that inherits from BaseRetriever
class CustomPineconeRetriever(BaseRetriever):
    pinecone_index: Any = Field(description="Pinecone index")
    embedding_function: Any = Field(description="Embedding function")
    namespace: str = Field(description="Namespace")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate embeddings for the query
        query_embedding = self.embedding_function.embed_query(query)
        
        # Query Pinecone
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            namespace=self.namespace,
            include_metadata=True
        )
        
        # Format results as LangChain documents
        docs = []
        for match in results.matches:
            metadata = match.metadata
            text = metadata.pop('text', '')
            docs.append(Document(page_content=text, metadata=metadata))
        
        return docs


class RAGChatbot:
    def __init__(self):
        # Initialize Gemini API key (from Streamlit secrets or environment)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        # Initialize Pinecone (from Streamlit secrets or environment)
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
        self.pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # PDF documents to download (relevant to ML, Python, and fraud detection)
        self.pdf_urls = [
            "https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf",
            "https://pandas.pydata.org/docs/pandas.pdf",
            "https://buildmedia.readthedocs.org/media/pdf/streamlit/stable/streamlit.pdf",
            "https://faculty.washington.edu/mdecock/papers/pmlr-v72-lopez-rojas18a.pdf",
            "https://www.aaai.org/Papers/KDD/1998/KDD98-011.pdf"
        ]

        # Initialize components to None (lazy loading)
        self.vector_store = None
        self.custom_retriever = None
        self.conversation_chain = None
        self.index_name = "credit-card-fraud-chatbot"
        self.use_custom_retriever = False
        
    def download_pdf(self, url, output_path):
        """Download a PDF from URL to a local file."""
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
        """Extract text from PDF using PyMuPDF."""
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    def setup_pinecone(self):
        """Initialize Pinecone and create index if needed."""
        # Check if index exists, otherwise create it
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,  
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for the index to be ready before proceeding
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        
    def process_pdfs(self):
        """Process PDFs and store them in Pinecone."""
        # Setup text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            all_documents = []

            # Download and process each PDF
            for i, url in enumerate(self.pdf_urls):
                pdf_path = os.path.join(temp_dir, f"document_{i}.pdf")

                # Download PDF
                success = self.download_pdf(url, pdf_path)
                if not success:
                    continue

                # Extract text
                documents = self.extract_text_from_pdf(pdf_path)

                # Split text into chunks
                chunks = text_splitter.split_documents(documents)
                all_documents.extend(chunks)

            # Initialize Pinecone
            self.setup_pinecone()

            # Use direct Pinecone approach
            # Create embeddings manually
            texts = [doc.page_content for doc in all_documents]
            metadatas = [doc.metadata for doc in all_documents]
            
            # Get embeddings
            embeddings_list = embeddings.embed_documents(texts)
            
            # Initialize Pinecone index directly
            index = self.pc.Index(self.index_name)
            
            # Insert data
            vectors = []
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings_list)):
                vectors.append({
                    'id': f'doc_{i}',
                    'values': embedding,
                    'metadata': {**metadata, 'text': text}
                })
            
            # Batch insert vectors
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch, namespace="credit-card-fraud-docs")
            
            # Create a custom retriever
            self.custom_retriever = CustomPineconeRetriever(
                pinecone_index=index,
                embedding_function=embeddings,
                namespace="credit-card-fraud-docs"
            )
            
            # Set flag to use custom retriever
            self.use_custom_retriever = True
            return self.custom_retriever

    def initialize_chain(self):
        """Initialize the conversational retrieval chain."""
        if not self.custom_retriever:
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name in existing_indexes:
                # Create a custom retriever directly
                index = self.pc.Index(self.index_name)
                
                # Create custom retriever
                self.custom_retriever = CustomPineconeRetriever(
                    pinecone_index=index,
                    embedding_function=embeddings,
                    namespace="credit-card-fraud-docs"
                )
                self.use_custom_retriever = True
            else:
                # If index doesn't exist, process PDFs
                self.process_pdfs()

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create chain with our custom retriever
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
        """Ask a question to the chatbot."""
        if not self.conversation_chain:
            self.initialize_chain()

        if not self.conversation_chain:
            return "Failed to initialize the chatbot. Please check the API keys and try again."

        try:
            response = self.conversation_chain({"question": question})
            return response["answer"]
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error: {str(e)}"


# Streamlit UI
def main():
    st.title("Credit Card Fraud Detection RAG Chatbot")
    st.write("Ask questions about credit card fraud detection, machine learning, or the Python libraries used for data analysis.")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize RAG chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # Chat input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Test the chatbot
if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = "AIzaSyBDbgnmAo705WKzXtGQVykpnxJtXUpUQdk"  
    os.environ["PINECONE_API_KEY"] = "pcsk_7PnniW_AZxLcRnGLGvRNdVJL6xqHUWqZD5KwfFyfSq5K1PFb6RD7PPw2HFrjZqTes2F7Ug"
    os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"
    
    # For direct testing
    #chatbot = RAGChatbot()
    #response = chatbot.ask("what is credit card fraud detection?")
    #print(response)
    main()