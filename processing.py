from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

def process_pdf(file_path, groq_api_key, google_api_key):
    # PDF Partitioning
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    # Separate elements
    tables = []
    texts = []
    for chunk in chunks:
        if "text_as_html" in vars(chunk.metadata):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    # Get images
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64

    images = get_images_base64(chunks)

    # Create summarization chains
    text_chain = create_text_chain(groq_api_key)
    table_chain= create_table_chain(google_api_key)
    image_chain = create_image_chain(google_api_key)

    # Generate summaries
    text_summaries = text_chain.batch(texts, {"max_concurrency": 3})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = table_chain.batch(tables_html, {"max_concurrency": 3})
    image_summaries = image_chain.batch(images)

    return {
        "texts": texts,
        "tables": tables_html,
        "images": images,
        "text_summaries": text_summaries,
        "table_summaries": table_summaries,
        "image_summaries": image_summaries
    }

def create_text_chain(api_key):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-guard-3-8b", api_key=api_key)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def create_table_chain(api_key):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024, google_api_key=api_key)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def create_image_chain(api_key):
    prompt_template = """Provide a **detailed and structured description** of the image, as it will serve as **contextual input** for a **multimodal Retrieval-Augmented Generation (RAG) system**.

    **Context:**
    The image is part of a research paper discussing **Transformer architectures**. Your description should be **precise, informative, and structured**, ensuring that all critical elements are captured.

    ### **Guidelines for Description:**
    - Identify and describe any **graphs, charts, or plots** (e.g., bar charts, line graphs, attention maps). Include information on their axes, labels, trends, and key observations.
    - Describe **mathematical notations, equations, or annotations** present in the image.
    - If the image contains **diagrams** (e.g., model architecture, flowcharts), break down its components and their relationships.
    - Mention **any text, labels, or legends** that provide crucial insights.
    - Explain **the purpose and significance** of the visual elements in relation to Transformers.
    - Use **precise technical terminology** while keeping the description coherent and structured.

    ### **Example Output Format:**
    - **Title/Caption (if available):** [...]
    - **Graph/Plot Details:** [...]
    - **Diagram Breakdown:** [...]
    - **Key Observations:** [...]
    - **Relevance to Transformers:** [...]

    Ensure the response is **well-structured and optimized for embeddings** in a multimodal RAG system."""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024, google_api_key=api_key)
    return prompt | model | StrOutputParser()