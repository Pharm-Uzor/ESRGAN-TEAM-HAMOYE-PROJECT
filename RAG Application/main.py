from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

import re
import io
import contextlib
import os
import time
from config import ANTHROPIC_API_KEY, GOOGLE_API_KEY, VOYAGE_API_KEY

anthropic_api_key = ANTHROPIC_API_KEY
google_api_key = GOOGLE_API_KEY
voyage_api_key = VOYAGE_API_KEY


gemini_embedding = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001', google_api_key=google_api_key)
voyager_embedding_model = VoyageEmbeddings(model = 'voyage-2', voyage_api_key=voyage_api_key)
def choose_model(model_name, temperature, top_p):
    model_dict = {'Claude-3.5-Sonnet':'claude-3-5-sonnet-20240620', 'Claude-3-Haiku':'claude-3-haiku-20240307',
                   'Claude-3-Opus':'claude-3-opus-20240229'}
    model = model_dict[model_name]
    return ChatAnthropic(model= model, temperature= temperature, top_p= top_p, max_tokens = 4096, anthropic_api_key=anthropic_api_key)

def get_retriever(file):
    file_bytes = file.read()
    with open(file.name, 'wb') as temp_file:
        temp_file.write(file_bytes)
    loader = PyPDFLoader(temp_file.name)
    docs = loader.load()
    docs1 = [doc.page_content for doc in docs]
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=2000, chunk_overlap=20,
                                                    length_function=len, is_separator_regex=False,)
    st.session_state.docs2 = text_splitter.create_documents(docs1)
    vectordb = FAISS.from_documents(st.session_state.docs2, gemini_embedding)
    st.session_state.retriever = vectordb.as_retriever()
    return st.session_state.retriever


def get_history_aware_retriever(retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.")
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever

system_prompt =   ( '''You are an expert data scientist skilled in seaborn, plotly, numpy, and pandas. Use the given data context to answer the user's question. 
            The data includes the following columns:
            - Pupil Per Teacher Ratio: indicates the number of students per teacher for a given country.
            - Government Expenditure on education (%): indicates the percentage of a government's total budget allocated to education.
            - Annual Teacher Salary (USD): indicates the average annual salary of teachers in US dollars.
            - Shortage of Learning Materials: an index indicating the extent to which students and teachers lack learning tools.More negative values indicate sufficiency of tools while more poistive values indicate insufficiency.
            - PISA Scores: obtained by averaging reading, math, science scores. This evaluates students reading, maths and science knowledge and skills.
            
            If you don’t know the answer, say so.

            Instructions:
            Analyze the Question:
            -Python Code: If the question involves data manipulation, calculations, or filtering, generate Python code.
            -Textual Response: If it requires summarizing data or describing trends, generate a textual response.

            Generate Response:
            Python Code response:
            -The Python code should be executable within Streamlit and potentially leverage libraries like pandas for data manipulation and visualization libraries like seaborn, matplotlib or plotly for creating charts.
            -Create executable Python code based on the question.
            -Do not load CSV files; use provided data directly.
            -Set country names as the dataframe index.
            -Render results such as texts, code, dataframes, graphs/tables in Streamlit.
            -To avoid parsing errors, do not provide any explanations or comments on your python code or your thought process
            -Always enclose outputs in triple quotations.
            -Use subplots for multiple plots.
            -Do not explain Python code
            
            Textual Response:
            -Provide a clear, concise text response if Python code is not suitable.
            -Render results in streamlit using either st.write or st.markdown
            

            Important Note:
            -I cannot access the data directly, so the code should use the provided data context (if available) within Streamlit.
            -Avoid generating code that loads external CSV files; focus on using the provided data.
            -Do not include explanatory text before or after the code.
            -You can include any relevant domain knowledge about factors impacting learning outcomes here (e.g., teacher quality, socioeconomic status).
           
            \'''
            import streamlit as st
            # ... (rest of your Python code for data analysis or visualization)

            # Assuming you have Python code or textual response generated

            # Display and execute the Python code (if applicable)
            st.write('**Python Code:**') 
            st.write(exec(python_code))

            # Display the textual response (if no Python code)
            st.write("**Textual Response:**")
            st.markdown(textual_response)  # Replace 'textual_response' with the generated text

            # Display visualizations (if applicable)
            # Use Streamlit charts (st.bar_chart, st.line_chart, etc.) or libraries like Plotly
            st.write("**Visualization:**")
            # ... (Your visualization code using streamlit or Plotly)
            \'''

            Context:
            {context}
                        ''')


def get_convo_rag_chain(system_prompt, history_aware_retriever):
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                  MessagesPlaceholder("chat_history"), ("human", "{input}"),])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",)
    return conversational_rag_chain


def clean_text(response):
    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    full_code = '\n'.join(code_blocks)
    return full_code

def exec_and_capture(code):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if 'import' in code or 'python' in code or 'streamlit' in code:
            st.code(code)
            exec(clean_text(code))
        else:
            st.write(code)
    return output.getvalue()

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def display_chat_history():
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if message['role'] == 'human':
            st.write(f"**You**: {message['content']}")
        else:
            st.write(f"**AI**: {message['content']}")
        st.write("---")  # Add a separator between messages

def display_performance_metric():
    st.subheader("Model Performance Metric")
    if st.session_state.performance_metric:
        for i, metric in enumerate(st.session_state.performance_metric):
            st.write(f"Query {i+1}:")
            st.write(f"Response Time: {metric['response_time']:.2f} seconds")
            st.write("---")
    else:
        st.write("No performance metric available yet.")

def invoke_llm(human_question, convo_rag_chain):
    start_time = time.time()
    progress_bar = st.progress(0)
    response = convo_rag_chain.invoke({"input": human_question}, 
                                      config={"configurable": {"session_id": "abc123"} },)["answer"]
    progress_bar.progress(20)  # Update progress after LLM call
    try:
        progress_bar.progress(50)  # Update progress before code execution
        exec_and_capture(response)
        progress_bar.progress(100)  # Update progress after code execution
    except Exception as e:
      st.error(f'Error: {e}')
      progress_bar.progress(100) # Update progress even on error
    end_time = time.time()
    response_time = end_time - start_time
    if 'import' in response or 'python' in response or 'streamlit' in response:
        response = clean_text(response)
    return {'response': response, 'response_time': response_time}

st.title('ESRGAN TEAM\'S RAG AI ASSISTANT')

if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
if 'performance_metric' not in st.session_state:
        st.session_state.performance_metric = []
if 'show_metric' not in st.session_state:
        st.session_state.show_metric = False
if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'history_aware_retriever' not in st.session_state:
    st.session_state.history_aware_retriever = None
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None

with st.sidebar:
    file = st.file_uploader('*Upload PDF Document*')
    model_name = st.radio('choose a model:', ('Claude-3.5-Sonnet', 'Claude-3-Haiku', 'Claude-3-Opus'))
    temperature = st.slider('choose a temperature:', 0.0, 1.0, 0.0, step = 0.1)
    top_p = st.slider('choose a Top_P Value:', 0.1, 1.0, 0.1, step = 0.1)
    llm = choose_model(model_name, temperature, top_p)
    if st.button('Reset Application'):
        reset_app()
    st.session_state.show_metric = st.checkbox("Show Model Performance Metric", value=st.session_state.show_metric)
    st.session_state.show_chat = st.checkbox("Show Chat History", value=st.session_state.show_chat)

if file:
    if st.session_state.retriever is None:
        st.session_state.retriever = get_retriever(file)
    if st.session_state.history_aware_retriever is None:
        st.session_state.history_aware_retriever = get_history_aware_retriever(st.session_state.retriever)
    if st.session_state.conversational_rag_chain is None:
        st.session_state.conversational_rag_chain = get_convo_rag_chain(system_prompt, st.session_state.history_aware_retriever)
    
    human_question = st.text_area('**SEND A MESSAGE**')
    if st.button('RUN'):
        if human_question:
            st.session_state.chat_history.append({"role": "human", "content": human_question})
            response = invoke_llm(human_question, st.session_state.conversational_rag_chain)
            st.session_state.chat_history.append({"role": "ai", "content": response["response"]})
            st.session_state.performance_metric.append({"response_time": response["response_time"],})

    if st.session_state.show_chat:  
        display_chat_history()
    if st.session_state.show_metric:
        display_performance_metric()
else:
    st.write("Please upload a PDF file.")
 

