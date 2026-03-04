import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


st.set_page_config(
    page_title="GenAI PDF Question Answering App",
    page_icon="📄",
    layout="wide"
)

st.title("📄 GenAI PDF Question Answering App")
st.write("Upload a PDF and ask questions about its content.")


# Load LLM once
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm()


uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    st.success("PDF uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name


    loader = PyPDFLoader(temp_path)
    documents = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = FAISS.from_documents(docs, embeddings)


    query = st.text_input("Ask a question about the document")

    if query:

        with st.spinner("Searching document..."):

            results = vectorstore.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""
            Use the following context to answer the question.

            Context:
            {context}

            Question:
            {query}
            """

            answer = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(answer)