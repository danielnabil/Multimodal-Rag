from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import  HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from base64 import b64decode
from bs4 import BeautifulSoup
import uuid

def setup_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries):
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    retriever.vectorstore.add_documents([
        Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) 
        for i, summary in enumerate(text_summaries)
    ])
    retriever.docstore.mset(zip(doc_ids, texts))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    retriever.vectorstore.add_documents([
        Document(page_content=summary, metadata={"doc_id": table_ids[i]})
        for i, summary in enumerate(table_summaries)
    ])
    retriever.docstore.mset(zip(table_ids, tables))

    # Add images
    img_ids = [str(uuid.uuid4()) for _ in images]
    retriever.vectorstore.add_documents([
        Document(page_content=summary, metadata={"doc_id": img_ids[i]})
        for i, summary in enumerate(image_summaries)
    ])
    retriever.docstore.mset(zip(img_ids, images))

    return retriever

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

def get_chain(retriever):

    
    # Parse docs function
    def parse_docs(docs):
        """Split base64-encoded images, texts, and extract tables as HTML."""
        b64 = []
        text = []
        tables = []

        for doc in docs:
            # Ensure doc is a string before performing string operations
            if not isinstance(doc, str):
                text.append(doc)  # Convert non-string elements to text
                continue  # Skip further processing for non-string elements

            # Check if the document contains an HTML table
            if "<table" in doc and "</table>" in doc:
                soup = BeautifulSoup(doc, "html.parser")
                table = soup.find("table")
                if table:
                    tables.append(str(table))  # Keep the table as raw HTML
                    continue  # Skip further processing if it's a table
            # Check if the document is an image (base64 encoded)
            try:
                b64decode(doc)
                b64.append(doc)
                continue  # Skip further processing if it's an image
            except Exception:
                pass  # Not a base64 image, continue processing



        return {"images": b64, "texts": text, "tables": tables}

    # Build prompt function
    def build_prompt(kwargs):

        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        context_tables = ""
        if len(docs_by_type["tables"]) > 0:
            for text_element in docs_by_type["tables"]:
                context_tables += text_element

        # construct prompt with context (including images)
        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Tables: {context_tables}
        Question: {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )
    return {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
    )