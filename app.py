from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_RAG_API_KEY")

file_path="Research_paper_chatbot/BTP_Final_Report (1).pdf"

loader = PyPDFLoader(file_path)
docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

vector_store = InMemoryVectorStore(hf)

document_ids = vector_store.add_documents(documents=all_splits)

from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template("""
You'll answer taking the context into consideration and your memory when the user inputs something. 
Think step by step before answering the question. 
The context is: {context}                              
User Question: {query}
 """)

llm=ChatMistralAI(model="mistral-small")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str



def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"query": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def classify(state: State):
    prompt=ChatPromptTemplate.from_template("""
    You'll answer taking the context into consideration and your memory when the user inputs something. 
    Think step by step before answering the question. 
    The context is: {context}                              
    User Question: {query}
    """)





if 'app' not in st.session_state:
    workflow = StateGraph(State)
    workflow.add_edge(START, "retrieve")
    workflow.add_node("retrieve", retrieve)
    workflow.add_edge("retrieve","generate")
    workflow.add_node("generate", generate)    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    st.session_state['app'] = app



st.title("RAG Chatbot with Langchain Memory")
input_text = st.text_input("Enter your prompt here")

# Consistent thread id for this user/session
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = "langchain_memory_example_1"

app = st.session_state['app']
thread_id = st.session_state['thread_id']

if input_text:
    # --- HIGHLIGHT: Pass only new user message and rely on Langchain memory checkpointer ---
    output = app.invoke(
        {"question": input_text}, 
        {"configurable": {"thread_id": thread_id}}
    )
    st.write(output["answer"])





