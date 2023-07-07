from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import streamlit as st
from streamlit_chat import message

#directory = '/content/'
#directory='/home/growlt194/Downloads/Twilio'
directory='/home/growlt194/Downloads/demo_paper'

#directory content loader

def load_docs(directory):
    loader = DirectoryLoader(directory,glob='**/*.pdf')
    documents = loader.load()
    print(documents)
    return documents


documents = load_docs(directory)
len(documents)


#doc splitter using tiktoken
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)
print(len(docs))


#document fetching from pinecone
api_key = "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"
embeddings = OpenAIEmbeddings(openai_api_key= api_key)


pinecone.init(api_key="5d04cef2-5aea-4abb-ba86-bf14cc06a398",environment="asia-southeast1-gcp-free")
index_name = "gpt-testcase"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


def get_similiar_docs(query, k=4, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs


model_name1 = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name = model_name1, openai_api_key=api_key )
#chain = load_qa_chain(llm, chain_type="stuff")

chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=api_key),retriever=index.as_retriever())


#def get_answer(query):
#    similar_docs = get_similiar_docs(query)
#    answer = chain.run(input_documents=similar_docs, question=query)
#    return answer


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = [] 

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()
 

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your pdf data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")