import os
import sys
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from tempfile import NamedTemporaryFile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pinecone

# initialize API
OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "langchain2"
global genre
global uploaded_file
def slidebarFunc():
    global genre
    global uploaded_file
    with st.sidebar:
        genre = st.radio(
            "Please select one",
            ('URL', 'Upload File'))
        if genre == 'URL':
            with st.sidebar.form(key='Form1'):
                search1 = st.text_input(label='Enter a URL link first')
                submit_button1 = st.form_submit_button(label='Enter')
                st.info(
                    """Enter online sites such as: \n- https://open.umn.edu \n- https://arxiv.org\n- https://en.wikipedia.org""")

            if submit_button1:
                # st.write(uploaded_file)
                # st.write(uploaded_file.name)
                try:
                    loader = UnstructuredURLLoader(urls=[search1])
                except:
                    loader = None
                    st.warning("Link is broken")
                return loader
        else:
            uploaded_file = st.file_uploader("Choose File", type=['pdf', 'txt'], accept_multiple_files=False, key=None, help=None,
                                             on_change=None, label_visibility="visible")
            # uploaded_file = st.file_uploader("File upload", type='csv')
            if uploaded_file is not None:
                if uploaded_file.name not in os.listdir("data"):
                    with open("data/" + uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("File Uploaded Successfully")
                loader = UnstructuredFileLoader("data/" + uploaded_file.name)

                return loader
            else:
                return None

st.set_page_config(page_title="Doculogue", page_icon="📄")
st.title("Doculogue")
loader = slidebarFunc()
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
    st.write(f'Now you have {len(texts)} documents')
    st.session_state["docsearch"] = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

if "hist" not in st.session_state:
    st.session_state["hist"] = []

template = """You are an AI assistant for answering questions about the Document you have uploaded.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
At the end of your answer, add a newline and return a python list of up to three URL sources which are related to the context
and question leading with a "#" like this without mentioning anything else:
$['topicURL1', 'topicURL2', 'topicURL3']

If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

=========
Context: {context}
=========
{chat_history}
User: {human_input}
AI Assistant:
Answer in Markdown:"""
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain

# memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history")
# llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

# memory=ConversationBufferWindowMemory(k=5, ai_prefix="AI Assistant")
tab1, tab2 = st.tabs(["Q&A", "History"])
with tab1:
    st.header("Q&A")
    form = st.form(key='my_form2')
    search = form.text_input(label='Search anything within the provided content')
    submit_button = form.form_submit_button(label='Enter')

    if submit_button:
        llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

        prompt_template = PromptTemplate(input_variables=["human_input", "context", "chat_history"], template=template)
        memory = ConversationBufferWindowMemory(k=8, return_messages=True, memory_key="chat_history",
                                               input_key="human_input")
        chain = load_qa_chain(llm=llm, chain_type="stuff", memory=memory, prompt=prompt_template)
        # chain = load_qa_chain(llm, chain_type="stuff")
        docs = st.session_state["docsearch"].similarity_search(search, include_metadata=True)
        # conversation = ConversationChain(llm=llm, chat_history=memory.load_memory_variables(inputs=[])['history'], prompt=prompt_template, context=docs, question=search)
        # chain = LLMChain(llm=llm, prompt=prompt_template, )
        # answer = chain.run(context=docs, question=search, chat_history=memory)
        answer = chain({"input_documents": docs, "human_input": search}, return_only_outputs=True)

        # answer = conversation(search)["response"]
        # bufw_history = conversation.memory.load_memory_variables(inputs=[])['history']
        # st.write(bufw_history)
        # st.write(answer)
        chain.memory.save_context({"human_input": f"{search}"}, {"output": f"{answer}"})
        st.session_state["hist"].append({"input": search, "output": answer["output_text"]})
        # st.write(st.session_state.hist)
        # st.info(answer.split("$")[0])

        st.info(answer["output_text"])
        # st.write(chain.memory.buffer)
        # st.write(chain.memory.load_memory_variables(inputs=[])['chat_history'])

with tab2:
    st.header("History")
    for index, key in enumerate(st.session_state['hist']):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(key["output"])
        with col2:
            st.info(key["input"])


