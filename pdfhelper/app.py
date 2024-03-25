from typing import List
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langserve import add_routes
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


# read pdf's from docs directory
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

def load_documents(directory: str) -> List[Document]:
    loader = PyPDFDirectoryLoader(directory)
    return loader.load_and_split()

def vector_store_retriever(vector_store: VectorStore, docs: List[Document], embedding: Embeddings) -> VectorStoreRetriever:
    return vector_store.from_documents(docs, embedding).as_retriever()


def chat_prompt_template(template: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(template)

chain = (
    {"context": vector_store_retriever(FAISS, load_documents("docs"), OpenAIEmbeddings()), "question": RunnablePassthrough()}
    | chat_prompt_template(template)
    | ChatOpenAI()
    | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Pdf reader"
)

add_routes(
    app,
    chain,
    path="/askpdf"
)