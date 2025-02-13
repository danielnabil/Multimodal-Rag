import streamlit as st
from processing import process_pdf
from chains import setup_retriever, get_chain
from utils import decode_base64_image
import tempfile

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📄 Training Reproducible DL Models Q&A Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
process_btn = st.button("Process Document")

if process_btn and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    with st.spinner("Processing document..."):
        processed_data = process_pdf(
            file_path,
            groq_api_key=st.secrets["GROQ_API_KEY"],
            google_api_key=st.secrets["GOOGLE_API_KEY"]
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