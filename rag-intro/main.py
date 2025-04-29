import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Function to use in custom rag implimentation
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("retrieving..")
    llm_model = os.getenv("MODEL_NAME_TO_USE")
    api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        api_key=api_key,
        temperature=0.1,
    )

    query = "What is a vector database?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result)

    vectorstore = PineconeVectorStore(
        index_name = os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrival_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})
    print("\n\n\n\n\n")
    print(result)
    

    ##
    ## Check waht happening behind the sean of the langchain functon through writing it.
    ##
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {Question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        { "context": vectorstore.as_retriever() | format_docs, "Question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    custom_res = rag_chain.invoke(query)
    print("\n\n\n\n\n")
    print(custom_res)