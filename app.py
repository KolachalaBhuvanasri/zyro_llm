import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼"
)

st.title("💼 Zyro Dynamics HR Help Desk")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

@st.cache_resource
def load_rag():

    loader = PyPDFDirectoryLoader(".")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        '''
You are the Zyro Dynamics HR Help Desk Assistant.

Answer ONLY using the provided HR policy context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, respond exactly:

I can only answer questions based on Zyro Dynamics HR policy documents.
'''
    )

    return retriever, llm, prompt

retriever, llm, prompt = load_rag()

question = st.chat_input("Ask an HR question...")

if question:

    docs = retriever.invoke(question)

    if len(docs) == 0:
        st.write(
            "I can only answer questions based on Zyro Dynamics HR policy documents."
        )
        st.stop()

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    st.write(answer)

    with st.expander("Sources"):

        for doc in docs:

            st.write(
                doc.metadata.get(
                    "source",
                    "Unknown Source"
                )
            )