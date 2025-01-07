# Ask-Your-File
This repository provides a Streamlit-based application that allows users to upload PDF, CSV, or WebPage documents, ask questions and the application will provide the answer from the uploaded file. The system processes the uploaded files, generates embeddings, and retrieves relevant information using LangChain and HuggingFace tools.

## Workflow
- Select the file type (PDF, CSV, or WebPage).
- Upload a document or enter a URL.
- The application processes and stores document embeddings using FAISS.
- Enter a question, and the system retrieves the most relevant answer.

<p align="center">
<image src="static/ask your file 1.png">
</p>

## Features
- Multi-Document Support: Works with PDFs, CSV files, and WebPages.
- LangChain Integration: Processes user queries and matches them with relevant information from documents.
- HuggingFace Embeddings: Generates document embeddings for accurate retrieval.
- Vector Search: Leverages FAISS for efficient document similarity search.
- User-Friendly Interface: Streamlit UI for easy interaction and querying.

## Tech Stack
- Language: Python
- Framework: Streamlit
- Embedding Model: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- LLM: ChatGroq (llama-3.1-70b-versatile)
- Vector Database: FAISS

## Installation:

### 1- Clone the Repository

```bash
git clone https://github.com/Ahmed-Farouk-Ibrahim/Ask-Your-File.git
cd Ask-Your-File  
```

### 2- Create a Conda Environment:
```bash
conda create -p venv python==3.10 -y
conda activate venv/
```

### 3- Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4- Configure Environment Variables:
- Create a `.env` file in the root directory with your API keys:

```ini
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Usage

### 1- Start the Application

```bash
streamlit run app.py  
```

### 2- Upload and Ask Questions
- Choose a file type (PDF, CSV, or WebPage) like here in the image:

<p align="center">
<image src="static/ask your file 1.png">
</p>

- Upload a file or provide a WebPage URL.
- Embed the document and ask your question.

### 3- View Results
- The response to your question will be displayed on the interface, below is an examble of how the application works:

<p align="center">
<image src="static/ask your file 2.png">
</p>

- Optionally, explore similar documents using the "Document Similarity Search" feature.

## Future Improvements:
- Add support for additional file formats (e.g., Word documents).
- Implement a summarization feature for large documents.
- Optimize processing time for larger datasets.

