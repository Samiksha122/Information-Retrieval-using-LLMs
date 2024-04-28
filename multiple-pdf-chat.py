import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
import openai
import os
import faiss
from langchain.document_loaders import PyPDFLoader
import time


openai.api_key  = 'enter-your-api-key-here'
openai.api_base = "https://limcheekin-mistral-7b-instruct-v0-1-gguf.hf.space/v1"
#pinecone.init(api_key='enter-your-api-key-here', environment='gcp-starter')


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator='\n',chunk_size=500,chunk_overlap=100)
    chunks=text_splitter.split_text(raw_text)
    return chunks


def get_vector_database(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    #index_name = "final"
    vectordatabase=FAISS.from_texts(text_chunks,embeddings)
    return vectordatabase

def get_prompt(system_prompt: str, instruction: str):
    prompt = f"<s>[INST]System: {system_prompt}[/INST]</s> [INST]User: {instruction}[/INST]"
    return prompt


def get_completion(prompt, system_prompt, temperature=0.1, max_tokens=512):
    messages = [{"role": "user", "content": get_prompt(system_prompt=system_prompt, instruction=prompt)}]
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=600,
        stream=True
    )
    return combine_chunks(start_time, response)

def combine_chunks(start_time, response):
    # create variables to collect the stream of chunks
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print(chunk_message.get('content', ''), end='')
        # print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

    # print the time delay and text received
    print(f"\nResponse time taken: {chunk_time:.2f}s")
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    return full_reply_content

def make_global(x):
    global vdb
    vdb=x

def handle_userinput(text):
    #global vectordb
    search_results=st.session_state.vectordb.similarity_search(text,k=6)
    top_relevant_text = ""

    for doc in search_results:
        # Extract the page content from each Document
        page_content = doc.page_content

        # Process and format the page content as needed
        formatted_chunk = page_content.strip()  # Remove leading/trailing whitespace if needed

        # Add the formatted chunk to the top relevant text with a paragraph break
        if top_relevant_text:
            top_relevant_text += "\n\n"  # Separate paragraphs with two newline characters
        top_relevant_text += formatted_chunk
        default_system_prompt="Answer the query based on the following paragraphs and if it seems that the query is unrelatd to the paragraph then reply as you would do normally without this paragraph" + top_relevant_text
    response=get_completion(text,default_system_prompt)
    st.write(search_results)
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    st.header("Chat with your PDFs :))")
    user_question=st.text_input("Ask a question from these PDFs")
    if user_question:

        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                #get_pdf_text
                raw_text=get_pdf_text(pdf_docs)

                #get_text_chunks
                text_chunks=get_text_chunks(raw_text)
                print(text_chunks)


                #get_vector database

                st.session_state.vectordb=get_vector_database(text_chunks)

                st.write("Processed Successfully !")
                #if user_question:
                #    handle_userinput(user_question, vector_database)

                #get conversation chain
                #st.session_state.conversation=get_conversation_chain(vectordb)

        #if user_question:
        #handle_userinput(user_question, vector_database)


if __name__=='__main__':
    main()

