import os

api_key = os.environ.get("GOOGLE_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

llm_model = os.getenv("MODEL_NAME_TO_USE")


if __name__ == "__main__":
    print("Hello vector")
    pdf_path = (
        "/home/abdhulla-k/aside/loading/ice_breaker/vectorstor-in-memory/react.pdf"
    )
    loader = PyPDFLoader(file_path=pdf_path)
    pages = loader.load()

    # Splitt the loaded document into managable chunks by llm
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    print(len(docs))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    vectorstore = FAISS.from_documents(documents=docs[30:33], embedding=embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrival_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        api_key=api_key,
        temperature=0.1,
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrival_qa_prompt)
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
