from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os


class OllamaModel:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")
        self.embed_model = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")
        self.vector_store = None
        self.message_history = []

    def save_text(self, user_text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(user_text)

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embed_model)
        else:
            self.vector_store.add_texts(chunks)

    def ask_question(self, user_input):
        self.message_history.append(user_input)
        context = "\n".join(self.message_history) + "\nOllama:"

        if self.vector_store is None:
            response = self.llm.invoke(context)
            return response

        retriever = self.vector_store.as_retriever()
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({"input": context})
        return response['answer']