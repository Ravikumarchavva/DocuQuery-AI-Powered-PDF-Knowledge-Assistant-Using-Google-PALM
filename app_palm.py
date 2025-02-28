import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()   
palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings(model="palm-1.5")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vector_store):
    embeddings = GooglePalmEmbeddings(model="palm-1.5")
    llm = GooglePalm()
    memory = ConversationBufferMemory(max_size=10)
    conversation_chain = ConversationalRetrievalChain(llm=llm,embeddings= embeddings , retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs before asking questions.")
        return
    
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"Bot: {message.content}")

def main():
    st.set_page_config("DocQuery: AI-Powerd PDF Knowledge Assistant")
    st.header("DocQuery: AI-Powerd PDF Knowledge Assistant")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF first.")
            else:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done! You can now ask questions about your documents.")
    
    user_question = st.text_input("Ask a question from PDF:")
    if user_question:
        user_input(user_question)
    
    # Display instruction message if no conversation exists
    if st.session_state.conversation is None:
        st.info("👆 Please upload your PDF documents and click 'Process PDFs' to start asking questions.")

if __name__ == "__main__":
    main()