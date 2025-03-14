import streamlit as st
from processing import process_pdf
from chains import setup_retriever, get_chain
from utils import decode_base64_image
import tempfile
import sqlite3
import os
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

required_secrets = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "UNSTRUCTURED_API_KEY"]
missing_secrets = [key for key in required_secrets if key not in st.secrets]

if missing_secrets:
    st.error(f"Missing secrets: {', '.join(missing_secrets)}")
    st.stop()
# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📄 Training Reproducible DL Models Q&A Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
process_btn = st.button("Process Document")

if process_btn and uploaded_file:
    if not os.path.exists('./docs/'):
        os.mkdir('./docs/')
    bytes_data = uploaded_file.read()
    file_path = os.path.join('./docs/', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(bytes_data)
    print(file_path)
    print("hereeeeee")

    with st.spinner("Processing document..."):
        processed_data = process_pdf(
            file_path,
            groq_api_key=st.secrets["GROQ_API_KEY"],
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            unstructured_api_key=st.secrets["UNSTRUCTURED_API_KEY"],
            uploaded_file=uploaded_file
        )
        
        st.session_state.retriever = setup_retriever(
            texts=processed_data["texts"],
            tables=processed_data["tables"],
            images=processed_data["images"],
            text_summaries=processed_data["text_summaries"],
            table_summaries=processed_data["table_summaries"],
            image_summaries=processed_data["image_summaries"]
        )
        
        st.session_state.chain = get_chain(st.session_state.retriever)
    st.success("Document processed successfully!")

# Query interface
query = st.text_input("💬 Ask a Question:", placeholder="e.g., Explain self-attention in transformers")

if query and st.session_state.retriever:
    with st.spinner("Generating answer..."):
        response = st.session_state.chain.invoke(query)
        
        st.markdown(f"### 🤖 AI Response:\n\n{response['response']}")
        
        # Display references
        col1, col2, col3 = st.columns(3)
        with col1:
            if response['context']['texts']:
                st.subheader("📄 Referenced Texts")
                for text in response['context']['texts']:
                    st.markdown(f"```\n{text.text}\n```")
        
        with col2:
            if response['context']['images']:
                st.subheader("🖼️ Referenced Images")
                images = [decode_base64_image(img) for img in response['context']['images']]
                st.image(images, use_column_width=True)
        
        with col3:
            if response['context']['tables']:
                st.subheader("📊 Referenced Tables")
                for table in response['context']['tables']:
                    st.markdown(table, unsafe_allow_html=True)