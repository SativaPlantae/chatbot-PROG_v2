import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🔐 Chave da OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# 📄 Função para carregar documentos e QA
@st.cache_resource
def carregar_qa_chain():
    caminho_pdf = "40.pdf"
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documentos)

    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # ✅ Prompt com context + question
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um assistente especializado em licenciamento ambiental.

Utilize o contexto abaixo para responder de forma clara e objetiva à pergunta feita.

Caso a resposta não esteja explicitamente presente, mas possa ser inferida com segurança, forneça-a mesmo assim."

-------------------
{context}

Pergunta: {question}
Resposta:"""
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=500)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# 🌐 Interface do app
st.set_page_config(page_title="Chatbot Institucional - Sativa Plantae", page_icon="🤖")
st.title("🤖 CHATBOT PROG (TESTE AVULSO)")
st.markdown("Faça perguntas sobre o conteúdo da AD n° 43/2024 📄")

user_question = st.text_input("Digite sua pergunta sobre o documento:")

if user_question:
    with st.spinner("Consultando o modelo..."):
        try:
            qa_chain = carregar_qa_chain()
            resposta = qa_chain.run(user_question)
            st.markdown("#### 💬 Resposta:")
            st.write(resposta)
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
