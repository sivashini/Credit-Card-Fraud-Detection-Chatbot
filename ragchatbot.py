import os
import time
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
import numpy as np


# Create a custom retriever class that inherits from BaseRetriever
class CustomPineconeRetriever(BaseRetriever):
    pinecone_index: Any = Field(description="Pinecone index")
    embedding_function: Any = Field(description="Embedding function")
    namespace: str = Field(description="Namespace")
    target_dimension: int = Field(description="Target dimension for embeddings")

    def adapt_embedding(self, embedding, target_dimension):
        """Adapt embedding to match the target dimension."""
        current_dim = len(embedding)

        if current_dim == target_dimension:
            # No adaptation needed
            return embedding
        elif current_dim < target_dimension:
            # Pad with zeros
            padding = [0.0] * (target_dimension - current_dim)
            return embedding + padding
        else:
            # Truncate
            return embedding[:target_dimension]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate embeddings for the query
        query_embedding = self.embedding_function.embed_query(query)

        # Adapt embedding to match index dimension
        adapted_query_embedding = self.adapt_embedding(query_embedding, self.target_dimension)

        # Query Pinecone
        results = self.pinecone_index.query(
            vector=adapted_query_embedding,
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
            "https://arxiv.org/pdf/2006.04939.pdf",  # Deep learning for sports trajectory
            "https://arxiv.org/pdf/1912.12024.pdf",  # Sports video classification
            "https://www.tomsa.in/wp-content/uploads/2020/05/TOMSA_Sports-Equipment.pdf",  # Football passing prediction
            "https://www.peai.org/wp-content/uploads/2014/04/Intro-to-Athletics-Resources.pdf",  # Sports pose estimation
            "https://repositorio.usp.br/directbitstream/85aae85f-a548-4033-b897-0a303df42b9e/3187254.pdf",  # Player behavior modeling
        ]

        # Initialize components to None (lazy loading)
        self.vector_store = None
        self.custom_retriever = None
        self.conversation_chain = None
        self.index_name = "large-sports-data-chatbot"
        self.use_custom_retriever = False
        self.index_dimension = 768  # Default dimension

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
        """Set up Pinecone index, adapting to existing index dimensions."""
        try:
            # Get list of available indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name in existing_indexes:
                # Get dimension of existing index
                index_info = self.pc.describe_index(self.index_name)
                dimension = index_info.dimension
                print(f"Using existing index '{self.index_name}' with dimension {dimension}")
                # Store the dimension for later use
                self.index_dimension = dimension
                return
            else:
                # Check if we've hit the index limit
                try:
                    # Try to create an index with dimension 768 (default for GoogleGenerativeAIEmbeddings)
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    self.index_dimension = 768

                    # Wait for the index to be ready
                    while not self.pc.describe_index(self.index_name).status["ready"]:
                        time.sleep(1)
                except Exception as e:
                    if "You've reached the max serverless indexes" in str(e):
                        # We've hit the index limit, so let's use an existing index
                        if len(existing_indexes) > 0:
                            self.index_name = existing_indexes[0]
                            index_info = self.pc.describe_index(self.index_name)
                            dimension = index_info.dimension
                            print(f"Hit index limit. Using existing index '{self.index_name}' with dimension {dimension}")
                            self.index_dimension = dimension
                        else:
                            raise Exception("No indexes available and cannot create new ones due to plan limits")
                    else:
                        raise e
        except Exception as e:
            print(f"Error in setup_pinecone: {e}")
            # Try to use any available index as a fallback
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if existing_indexes:
                self.index_name = existing_indexes[0]
                index_info = self.pc.describe_index(self.index_name)
                self.index_dimension = index_info.dimension
                print(f"Using fallback index '{self.index_name}' with dimension {self.index_dimension}")
            else:
                raise Exception("No indexes available to use")

    def adapt_embeddings(self, embeddings_list, target_dimension):
        """Adapt embeddings to match the target dimension."""
        adapted_embeddings = []

        for embedding in embeddings_list:
            current_dim = len(embedding)

            if current_dim == target_dimension:
                # No adaptation needed
                adapted_embeddings.append(embedding)
            elif current_dim < target_dimension:
                # Pad with zeros
                padding = [0.0] * (target_dimension - current_dim)
                adapted_embeddings.append(embedding + padding)
            else:
                # Truncate
                adapted_embeddings.append(embedding[:target_dimension])

        return adapted_embeddings

    def process_pdfs(self):
        """Process PDFs and store them in Pinecone."""
        # Initialize Pinecone first to know the target dimension
        self.setup_pinecone()
        target_dimension = self.index_dimension

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

            # Use direct Pinecone approach
            # Create embeddings manually
            texts = [doc.page_content for doc in all_documents]
            metadatas = [doc.metadata for doc in all_documents]

            # Get embeddings
            embeddings_list = embeddings.embed_documents(texts)

            # Adapt embeddings to match the target dimension
            adapted_embeddings = self.adapt_embeddings(embeddings_list, target_dimension)

            # Initialize Pinecone index directly
            index = self.pc.Index(self.index_name)

            # Insert data
            vectors = []
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, adapted_embeddings)):
                vectors.append({
                    'id': f'doc_{i}',
                    'values': embedding,
                    'metadata': {**metadata, 'text': text}
                })

            # Batch insert vectors
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch, namespace="large-sports-data-docs")

            # Create a custom retriever
            self.custom_retriever = CustomPineconeRetriever(
                pinecone_index=index,
                embedding_function=embeddings,
                namespace="large-sports-data-docs",
                target_dimension=target_dimension
            )

            # Set flag to use custom retriever
            self.use_custom_retriever = True
            return self.custom_retriever

    def initialize_chain(self):
        """Initialize the conversational retrieval chain."""
        if not self.custom_retriever:
            try:
                # Setup Pinecone to determine index and dimension
                self.setup_pinecone()
                target_dimension = self.index_dimension

                # Create embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.gemini_api_key
                )

                # Create a custom retriever
                index = self.pc.Index(self.index_name)
                self.custom_retriever = CustomPineconeRetriever(
                    pinecone_index=index,
                    embedding_function=embeddings,
                    namespace="large-sports-data-docs",
                    target_dimension=target_dimension
                )

                # Check if index has data
                # Try a simple query to see if there are vectors
                query_embedding = embeddings.embed_query("sports")
                adapted_query = self.adapt_embeddings([query_embedding], target_dimension)[0]
                result = index.query(vector=adapted_query, top_k=1, namespace="large-sports-data-docs")

                if not result.matches:
                    print("Index exists but has no data. Processing PDFs...")
                    self.process_pdfs()
                else:
                    print("Index has data. Using existing index.")
                    self.use_custom_retriever = True
            except Exception as e:
                print(f"Error checking index: {e}")
                # Process PDFs to create or populate index
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
            try:
                self.initialize_chain()
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return f"Error initializing chatbot: {str(e)}"

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
    st.title("Large Sports Data RAG Chatbot")
    st.write("Ask questions about sports data, performance analytics, player statistics, or related research.")

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
    os.environ["GEMINI_API_KEY"] = "AIzaSyBLa4Mcc6zueockkXLx2UliNkANQJQMz6I"
    os.environ["PINECONE_API_KEY"] = "pcsk_4JZgKo_QsiRNzXmrM4EHaHU6vXeGAPqeJvzQVA8uQkQ1gjNVfp9XbKCAGFDoBPqej7nvpy"
    os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

    # For direct testing
    #chatbot = RAGChatbot()
    #response = chatbot.ask("ios")
    #print(response)
    main()

