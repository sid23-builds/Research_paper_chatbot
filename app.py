from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from huggingface_hub import snapshot_download
from langgraph.graph import MessagesState, START, StateGraph, END
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_RAG_API_KEY")

file_path="/Users/sidharthjain/Desktop/Data Science Projects/Chatbot/RAG chatbot/BTP_Final_Report (1).pdf"

loader = PyPDFLoader(file_path)
docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
snapshot_download(repo_id="BAAI/bge-small-en")
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

# prompt=ChatPromptTemplate.from_template("""
# You'll answer taking the context into consideration and your memory when the user inputs something. 
# Think step by step before answering the question. 
# The context is: {context}                              
# User Question: {query}
#  """)

llm = ChatMistralAI(model="mistral-large-latest")

# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str
@tool
def retrieve(query: str)-> str:
    """Use this tool to answer ANY question about the research paper.
    This includes its topic, abstract, introduction, methods, results, 
    discussion, or any other detail. 
    If the user question contains words like 'paper', 'research', 'study', 
    'experiment', 'BTP', 'report', or any academic-related terms,
    ALWAYS call this tool instead of answering directly."""
    
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized


tools = ToolNode([retrieve])



def query_or_respond(state: MessagesState):
    """Call the tool if the input contains the word "research" """

    llm_with_tools = llm.bind_tools([retrieve])   
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
# print(response) gives the following output: content="I'm a helpful assistant, but I don't have personal knowledge or memories. I can't say for certain whether I know about your research paper or not, as I don't have the ability to recall specific interactions with individual users. However, I'm here to help with your research paper or any other questions you might have! Please feel free to share some details about your research paper and I'll do my best to provide useful and relevant information." additional_kwargs={} response_metadata={'token_usage': {'prompt_tokens': 16, 'total_tokens': 113, 'completion_tokens': 97}, 'model_name': 'mistral-small', 'model': 'mistral-small', 'finish_reason': 'stop'} id='run--45795304-f0fe-4f02-b9f7-009620fcbba2-0' usage_metadata={'input_tokens': 16, 'output_tokens': 97, 'total_tokens': 113}


def generate(state: MessagesState):
    
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    

    
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}





if 'graph' not in st.session_state:
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_edge(START, "query_or_respond")
    graph_builder.add_node("query_or_respond",query_or_respond)
    graph_builder.add_node("tools",tools)
    graph_builder.add_conditional_edges(
                "query_or_respond",
                tools_condition,
                {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_node("generate",generate)
    graph_builder.add_edge("generate", END)
    memory=MemorySaver()
    graph=graph_builder.compile(checkpointer=memory)
    st.session_state['graph'] = graph



st.title("RAG Chatbot with Langchain Memory")
input_text = st.text_input("Enter your prompt here")

config = {"configurable": {"thread_id": "abc234"}}

graph = st.session_state['graph']


if input_text:

    response=graph.invoke({"messages": [{"role": "user", "content": input_text}]}
                          , config)
    st.write(response["messages"][-1].content)






