import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message as chat_message
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate


# Load the OpenAI API key
# open_ai_key=st.secrets['OPENAI_API_KEY']

def main():

    # Set up the app title and header
    st.set_page_config(page_title="AskifyDocs")
    st.header("Askify Docs ")


    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        # Allow users to upload files
        uploaded_files= st.file_uploader("Upload your file", type=['pdf', 'docx'],accept_multiple_files=True)
        open_ai_key=st.text_input("OpenAI API Key", key="chatbot_api_key",type='password')

        # Load the OpenAI API key
        openai_api_key=open_ai_key

        # Get user-defined prompt template
        user_prompt=get_prompt_input()

        prcess= st.button("Process")

        if prcess:

            # Ensure all required inputs are provided
            if not openai_api_key:
                st.info("Please add your OpenAI API Key to continue....")
                st.stop()
            if not user_prompt:
                st.info("Please add your Prompt to continue....")
                st.stop()
            
            if not uploaded_files:
                st.info("Please Upload your file to continue....")
                st.stop()

            # Extract text from uploaded files
            files_text=get_text_from_files(uploaded_files)
            st.write("File Loaded")

            # Split the extracted text into smaller chunks
            chunks_of_text=get_chunks_of_text(files_text)
            st.write("File divide in chunks")

            # Create vector store
            vectorstore=get_vectorstore(chunks_of_text)
            st.write("Vector store Created")

            # Create a conversation chain using the vector store and prompt
            st.session_state.conversation = get_conversation_chain(vectorstore,openai_api_key,user_prompt)
        
            st.session_state.processComplete= True 

    if st.session_state.processComplete == True:

        # Input for user questions
        user_question= st.chat_input("Ask Question related to Uploaded file.")
        if user_question:
            handle_userinput(user_question)



# Function to get a custom prompt input from the user
def get_prompt_input():
    # Text area for the user to define their prompt template
    user_prompt = st.text_area(
        "Enter your custom prompt template:",
        value=(
            """ In start say Hi, how are you?  How can I assist you today?

                You are a helpful assistant. Start by asking the user a relevant question based on their input or request. If additional clarification is needed, ask follow-up questions to better understand the user's needs.

                Once you have enough context, answer the user's question strictly based on the provided document context. 
                If the answer is not in the context, respond with: "Sorry, I cannot answer that.
        """
        ),
        key="custom_prompt",
        height=150
    )

    return user_prompt

# Function to extract text from uploaded files
def get_text_from_files(uploaded_files):
    pdf_text= ""
    for uploaded_file in uploaded_files:

        # Extract file extension to determine file type
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            # Extract text from PDF pages
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                 pdf_text += page.extract_text()    
        else:
            print("File typr not supported")

    return pdf_text

# Function to split text into smaller chunks    
def get_chunks_of_text(files_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks=text_splitter.split_text(files_text)
    print("geting chunks text")

    return chunks

# Function to create a vector store for document search
def get_vectorstore(chunks_of_text):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks_of_text,embeddings)

    print("in vector function")
    return knowledge_base


# Function to set up a conversational retrieval chain
def get_conversation_chain(vectorstore,openai_api_key,user_prompt):
    # Initialize the OpenAI chat model
    llm = ChatOpenAI(openai_api_key= openai_api_key, model_name="gpt-3.5-turbo",temperature=0.5 )
    memory= ConversationBufferMemory(memory_key='chat_history', return_messages=True)


    # Create a custom prompt 
    full_prompt = f"""
    {user_prompt}

    Use the following context to answer the user's question:
    Context: {{context}}
    Question: {{question}}
    Answer:
    """

     # Use the user-defined prompt
    custom_prompt = PromptTemplate(
        template=full_prompt,
        input_variables=["context", "question"]
        
    )


    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        memory=memory 
    )
    print("in conversation chain function")

    return conversation_chain

# Function to handle user questions
def handle_userinput(user_question):
    with get_openai_callback() as cb:

        # Generate a response to the user's question
        response = st.session_state.conversation({'question' : user_question})
        
    # Update the chat history with the response   
    st.session_state.chat_history = response['chat_history']

    # Display chat history 
    response_container = st.container()
    with response_container:
        for i, chat in enumerate(st.session_state.chat_history):
            if i % 2 ==0:
                chat_message(chat.content, is_user=True, key =f"user_{i}")

            else:
                chat_message(chat.content,key = f"bot_{i}")
    print("in the user question function")



if __name__ == "__main__":
    main()