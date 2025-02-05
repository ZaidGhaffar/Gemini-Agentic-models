import pickle
import os
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class RAG:
    def __init__(self, file_path, api_key, save_path=r"C:\python\AI-Agents\app\db"):
        self.file_path = file_path
        self.api_key = api_key
        self.save_path = save_path
        self.faiss_index = None
        self.doc_texts = []

        # Configure Gemini AI
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in the .env file.")
        genai.configure(api_key=self.api_key)

    def document_loader(self):
        """Loads the PDF document."""
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        return docs

    def text_splitter(self, docs):
        """Splits documents into manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)

    def load_embed_model(self, data):
        """Generates embeddings using Gemini AI."""
        response = genai.embed_content(model="models/text-embedding-004", content=data)
        return np.array(response["embedding"]).astype(np.float32)

    def storing_embeddings(self, docs):
        """Stores document embeddings in FAISS and saves them."""
        texts = [doc.page_content for doc in docs]
        embeddings = np.array([self.load_embed_model(text) for text in texts])

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        self.doc_texts = texts  # Store mapping between texts and embeddings

        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{self.save_path}.index")

        # Save document texts
        with open(f"{self.save_path}_texts.pkl", "wb") as f:
            pickle.dump(self.doc_texts, f)

        print("Embeddings created and saved successfully! ✅")

    def load_embeddings(self):
        """Loads FAISS index and document texts if they exist."""
        try:
            self.faiss_index = faiss.read_index(f"{self.save_path}.index")

            with open(f"{self.save_path}_texts.pkl", "rb") as f:
                self.doc_texts = pickle.load(f)

            print("Embeddings loaded successfully! ✅")
        except Exception as e:
            print("No existing embeddings found. Please generate embeddings first.")

    def query_ans(self, query):
        """Finds relevant documents based on the user's query."""
        if self.faiss_index is None:
            print("Error: FAISS index not initialized. Please load or generate embeddings.")
            return None

        query_embedding = self.load_embed_model(query).reshape(1, -1)  # Ensure 2D shape
        top_k = 3
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        if len(indices[0]) == 0:
            print("No relevant documents found.")
            return None

        relevant_docs = [self.doc_texts[i] for i in indices[0]]
        context = "\n".join(relevant_docs)
        print(f"Retrieved Context:\n{context}")
        return context

if __name__ == "__main__":
    pass
    