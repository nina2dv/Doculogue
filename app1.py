import os
import sys
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import pinecone
st.set_page_config(page_title="Doculogue", page_icon="📄")

# initialize API
OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
pinecone_index = pinecone.Index('langchain2')
index_name = "langchain2"


global genre
global uploaded_file


def generate_namespace():
    placeholder = st.sidebar.empty()
    with placeholder:
        form1 = st.form("space")
        search3 = form1.text_input(label='Enter namespace first', help="This is where your"
                                                                       " documents embeddings will be stored")
        submit_button3 = form1.form_submit_button(label='Enter')
    if submit_button3:
        st.session_state["namespace"] = search3
        placeholder.empty()


def delete_namespace():
    with st.sidebar:
        submit_button4 = st.button(label='Clear vectors from namespace')
    if submit_button4:
        pinecone_index.delete(deleteAll=True, namespace=st.session_state["namespace"])


def search_docs(index, query):
    # Search for similar chunks
    docs = index.similarity_search(query, include_metadata=True)
    return docs


def slidebar_func():
    global genre
    global uploaded_file
    with st.sidebar:
        genre = st.radio(
            "Please select one",
            ('URL', 'Upload File'))
        if genre == 'URL':
            with st.sidebar.form(key='Form1'):
                search1 = st.text_input(label='Enter a URL link')
                submit_button1 = st.form_submit_button(label='Enter')
                st.info(
                    """Enter online sites such as: \n- https://open.umn.edu \n- https://arxiv.org\n- https://en.wikipedia.org""")

            if submit_button1:
                try:
                    loader = UnstructuredURLLoader(urls=[search1])
                except:
                    loader = None
                    st.warning("Link is broken")
                return loader
        elif genre == "Upload File":
            uploaded_file = st.file_uploader("Choose File", type=['pdf', 'txt'], accept_multiple_files=False, key=None, help=None,
                                             on_change=None, label_visibility="visible")
            if uploaded_file is not None:
                if uploaded_file.name not in os.listdir("data"):
                    with open("data/" + uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("File Uploaded Successfully")
                loader = UnstructuredFileLoader("data/" + uploaded_file.name)

                return loader
            else:
                return None


st.title("Doculogue")
if "namespace" not in st.session_state:
    generate_namespace()

if "namespace" in st.session_state:
    delete_namespace()

loader = slidebar_func()

if loader is not None:
    try:
        data = loader.load()
        if (genre == "Upload File"):
            dir = "data/"
            for file in os.scandir(dir):
                os.remove(file.path)
    except UnboundLocalError:
        st.warning("URL Request Error")
        sys.exit()
    # st.write(data)
    # st.write(f'You have {len(data)} document(s) in your data')
    # st.write(f'There are {len(data[0].page_content)} characters in your document')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    # st.write(f'Now you have {len(texts)} documents')
    try:
        Pinecone.from_texts([t.page_content for t in texts], embedding=embeddings, index_name=index_name,
                            namespace=st.session_state["namespace"])
        st.session_state["docsearch_namespace"] = Pinecone.from_existing_index("langchain2",embedding=embeddings,
                                                                               namespace=st.session_state["namespace"])
    except AttributeError:
        st.warning("Please enter a namespace")

if "hist" not in st.session_state:
    st.session_state["hist"] = []

template = """You are an AI assistant for answering questions about the Document you have uploaded.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
At the end of your answer, provide the sources used for your answer such as the page number and the name of the document.

If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

=========
{context}
=========
{chat_history}
User: {human_input}
AI Assistant:
Answer in Markdown:"""

memory = ConversationBufferWindowMemory(k=5, ai_prefix="AI Assistant")
tab1, tab2 = st.tabs(["Q&A", "History"])

prompt_template = PromptTemplate(input_variables=["human_input", "context", "chat_history"], template=template)
memory = ConversationBufferWindowMemory(k=8, return_messages=True, memory_key="chat_history",input_key="human_input")
chain = load_qa_chain(llm=OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
                          , chain_type="stuff", memory=memory, prompt=prompt_template)
def ask(user_input):
    docs = search_docs(st.session_state["docsearch_namespace"], user_input)
    answer = chain({"input_documents": docs, "human_input": user_input}, return_only_outputs=True)
    return answer

with tab1:
    st.header("Q&A")
    form = st.form(key='my_form2')
    search = form.text_input(label='Search anything within the provided content')
    submit_button = form.form_submit_button(label='Enter')

    if submit_button:
        response = ask(search)
        chain.memory.save_context({"human_input": f"{search}"}, {"output": f"{response}"})
        st.session_state["hist"].append({"input": search, "output": response["output_text"]})
        st.info(response["output_text"])

with tab2:
    st.header("History")
    for index, key in enumerate(st.session_state['hist']):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(key["output"])
        with col2:
            st.info(key["input"])
