import streamlit as st
from streamlit_chat import message
import pdfplumber  
import docx
import os
import tempfile
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_model = LlamaCpp(
        model_path="PATH OF YOUR QUANTIZED LLAMA2 MODEL",
        temperature=0.5,  # Level of creativity (controls the diversity of generated text)
        n_gpu_layers=40,  # Number of GPU-accelerated layers required for the Llama2 model
        n_batch=512,      # Amount of text processed at once (batch size)
        max_tokens=2000,  # Maximum number of tokens allowed for processing in a single inference step
        top_p=1,          # Top-p sampling probability threshold (controls diversity in text generation)
        callback_manager=callback_manager,  # Callback manager for handling callbacks during model execution
        verbose=True     # Whether to enable verbose mode for additional logging or output
    )
    return llama_model

# Load the model
llm = load_model()

# Session state allows you to store data persistently across Streamlit app reruns, which means the data will be preserved even when the app is rerun or refreshed.
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about uploaded document"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    # Tokenize the query
    tokens = query.split()  # Splitting by whitespace

    # Initialize variables to store chunks of tokens
    chunked_queries = []
    current_chunk = []

    # Loop through tokens and create chunks that fit within token limit
    for token in tokens:
        if len(current_chunk) + len(token) <= 512:
            current_chunk.append(token)
        else:
            # If adding the token would exceed token limit, store the current chunk
            chunked_queries.append(" ".join(current_chunk))
            current_chunk = [token]

    # Add the last chunk
    if current_chunk:
        chunked_queries.append(" ".join(current_chunk))

    # Initialize variable to store combined response
    combined_response = ""

    # Loop through chunked queries and get response for each chunk
    for chunk in chunked_queries:
        result = chain({"question": chunk, "chat_history": history})
        history.append((chunk, result["answer"]))
        combined_response += result["answer"] + " "

    return combined_response

def display_chat_history(chain):
    # Create containers for displaying chat history and user input
    reply_container = st.container()
    container = st.container()

    # Display user input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        # If user submits input and there is text entered
        if submit_button and user_input:
            with st.spinner('Generating response...'):
                # Generate response to user input using conversation_chat function
                output = conversation_chat(user_input, chain, st.session_state['history'])

            # Append user input to past conversation
            st.session_state['past'].append(user_input)
            # Append generated response to past conversation
            st.session_state['generated'].append(output)

    # If there are generated responses
    if st.session_state['generated']:
        # Display chat history
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(text_chunks):
    # Create embeddings
    # The function creates a tool called embeddings to represent text as numerical vectors. 
    # It uses the model sentence-transformers/all-MiniLM-L6-v2 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})

    # The function builds a library called vector_store using FAISS, a tool for efficient vector storage and retrieval.
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    
    # A ConversationBufferMemory component is created to keep track of the ongoing conversation history.
    # It stores previous questions and answers under the key "chat_history".
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # This chain acts as the backbone for the chatbot's ability to respond to queries and hold a coherent conversation.
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                  chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain


def calculate_cosine_similarity(reference_text, generated_text):
    vectorizer = CountVectorizer().fit_transform([reference_text, generated_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
    similarity_percentage = similarity[0][0] * 100  # Convert to percentage
    return similarity_percentage

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Multiple docs summarization", 
    page_icon="🐏"
)

def main():
        
    st.title("Multi-Docs Summarization using llama2 :books:")
    option = st.radio("Select Input Type", ("Text Input", "File Upload"))

    if option == "Text Input":
        user_input = st.text_area("Enter your prompt:")

        # Generate response when user clicks a button
        if st.button("Generate Response"):
            if user_input:
                # Split the input text into chunks
                input_chunks = [user_input[i:i+512] for i in range(0, len(user_input), 512)]
                # Generate response using the loaded model for each chunk
                response = ""
                for chunk in input_chunks:
                    response += llm(chunk)
                # Display the response
                st.write("Model Response:")
                st.write(response)
                # Calculate and display cosine similarity as a percentage
                similarity_percentage = calculate_cosine_similarity(user_input, response)
                st.write(f"Accuracy: {similarity_percentage:.2f}%")
                
            else:
                st.write("Please enter a prompt.")

    else:  # File Upload
        summary_uploaded_files = st.file_uploader("Upload multiple files", type=["txt", "pdf", "doc", "docx"], accept_multiple_files=True)
        if summary_uploaded_files:
            for uploaded_file in summary_uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == "pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text()
                elif file_extension == "docx":
                    # Read .docx file and extract text
                    doc = docx.Document(uploaded_file)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + '\n'

                else:
                    # For other file types (txt), read the content as bytes and decode
                    file_contents = uploaded_file.read()
                    text = file_contents.decode("utf-8")

                # Split the input text into chunks
                input_chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                # Generate summary using the loaded model for each chunk
                summary = ""
                for chunk in input_chunks:
                    summary += llm(chunk)
                # Display the summary for each document
                st.write("Summary for", uploaded_file.name)
                st.write(summary)
                
                # Calculate and display cosine similarity as a percentage
                similarity_percentage = calculate_cosine_similarity(text, summary)
                st.write(f"Accuracy: {similarity_percentage:.2f}%")



    # chat bot
    st.title("Multi-Docs ChatBot using llama2 :books:")
    initialize_session_state()

    # Initialize Streamlit
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in (".docx", ".doc"):
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create the chain object
        chain = create_conversational_chain(text_chunks)
        
        # Display chat history
        display_chat_history(chain)

if __name__ == "__main__":
    main()
