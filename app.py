from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return "Medical Chatbot Backend Running"

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")

    # Handle empty input
    if not message:
        return jsonify({"answer": "Please enter a valid query"})

    # Load embeddings ONLY during request
    embeddings = download_hugging_face_embeddings()

    # Connect Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Connect existing index
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    # Search
    docs = docsearch.similarity_search(message, k=2)

    if docs:
        response = docs[0].page_content.replace("\n", " ").strip()[:300]
    else:
        response = "No relevant information found."

    return jsonify({"answer": response})

# Run server
if __name__ == "__main__":
    app.run(debug=True)