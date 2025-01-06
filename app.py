import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
import tempfile
import pandas as pd

# Load environment variables
load_dotenv()

# Streamlit UI: Set up the Streamlit app
st.set_page_config(page_title="Ask Your File",
                   page_icon='✍️',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Ask Your File ✍️")

@st.cache_resource
def load_llm():
    return ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM and embeddings with caching
llm = load_llm()
embeddings = load_embeddings()

# Load the GROQ and OpenAI API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding(file_type, file_content):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings

        if file_type == "PDF":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            st.session_state.loader = PyPDFLoader(temp_file_path)

        elif file_type == "CSV":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            df = pd.read_csv(temp_file_path)
            rows = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).tolist()
            combined_docs = [' '.join(rows[i:i+10]) for i in range(0, len(rows), 10)]
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = [split for doc in combined_docs for split in st.session_state.text_splitter.split_text(doc)]

            chunks = [st.session_state.final_documents[i:i+50] for i in range(0, len(st.session_state.final_documents), 50)]
            all_embeddings = []
            for chunk in chunks:
                all_embeddings.extend(st.session_state.embeddings.embed_documents(chunk))
            
            st.session_state.vectors = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
            return

        elif file_type == "WebPage":
            st.session_state.loader = WebBaseLoader(file_content)

        # For PDF and WebPage, proceed with document loading and vector store creation
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

file_type = st.radio("Choose file type", ("PDF", "CSV", "WebPage"))

if file_type:
    st.session_state.pop("prompt_input", None)
    st.session_state.pop("vectors", None)
    st.session_state.pop("final_documents", None)
    st.session_state.pop("docs", None)

uploaded_file = None

if file_type in ["PDF", "CSV"]:
    uploaded_file = st.file_uploader(f"Upload your {file_type} file")
elif file_type == "WebPage":
    uploaded_file = st.text_input("Enter the WebPage URL")

question = st.text_input("Enter Your Question From Documents", value=st.session_state.get("prompt_input", ""))

if uploaded_file:
    vector_embedding(file_type, uploaded_file.getvalue() if file_type in ["PDF", "CSV"] else uploaded_file)
    #st.write("Vector Store DB Is Ready")

if question and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

    st.session_state.prompt_input = question
