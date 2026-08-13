from dotenv import load_dotenv 
load_dotenv()

from datasets import load_dataset

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

#loadintg the dataset [1000 only]
dataset = load_dataset("CShorten/ML-ArXiv-Papers")
papers=dataset["train"].select(range(1000))
print(dataset)
print('total papers : ', len(papers))

#convert dataset as documents of page_content and title
documents = []
for paper in papers:
    documents.append(
        Document(
            page_content=paper["abstract"],
            metadata={"title": paper["title"]}
            )
    )
print("documents : ", len(documents))

#split documents as chunks for embedding
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50
)
chunks=splitter.split_documents(documents)
print("chunks : ", len(chunks))

#embedding HF model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#pinecone store
vectorStore=PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model, 
    index_name="rag-index"
)
print("inserted into pinecone")

#retriever
retriever=vectorStore.as_retriever(
    search_kwargs={"k":5}
)

#local LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

#RAG prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a research assistant.
    Answer the question only using the context.
    context: {context}
    question: {question}
"""
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#LangChain Expression Language LCEL RAG chain
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()
    } | prompt | llm
)

# RunnablePassthrough() is a core component used to pass data through a pipeline unchanged or with added keys. 
# It acts as an identity function within the LangChain Expression Language (LCEL). 
# It allows you to preserve original user inputs alongside intermediate processed data

#ask question and get response retrieved
question= """"
    what is the methodology mentioned in this paper?
"""
response=rag_chain.invoke(question)

print("\n=======================")
print("ANSWER")
print("=========================")
print(response.content)
print("=========================")