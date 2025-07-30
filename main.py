from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = FastAPI()

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

class ChatRequest(BaseModel):
    prompt: str
    
class ChatResponse(BaseModel):
    answer: str
    context: List[str]
    
llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen/qwen3-32b")

# Define the prompt template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

@app.post("/chat", response_model=ChatResponse)
def get_response(request: ChatRequest):
    """
    Chat endpoint for generating a response based on the prompt and context.
    """
    try:
        search = TavilySearch(max_results=1)
        output = search.run(request.prompt)
        if not output:
            raise HTTPException(status_code=404, detail="No search results found.")
        url = output['results'][0]['url']
        if not url:
            raise HTTPException(status_code=404, detail="No URL found in search results.")
        # Load documents from a website
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        # Convert documents to vectors using embeddings
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({"input": request.prompt})
        print("Response time: ", time.process_time() - start)
        return ChatResponse(
            answer=response["answer"],
            context=[doc.page_content for doc in response["context"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.get("/documents", response_model=List[str])
# def get_documents():
#     """
#     Endpoint to get loaded document chunks (for debugging or exploration).
#     """
#     return [doc.page_content for doc in final_documents[:10]]

# @app.post("/reload_documents")
# def reload_documents():
#     """
#     Endpoint to reload the documents and embeddings.
#     """
# #     global docs, final_documents, vectors, retriever
#         try:
#             docs = loader.load()
#             final_documents = text_splitter.split_documents(docs)
#             vectors = FAISS.from_documents(final_documents, embeddings)
#             retriever = vectors.as_retriever()
#             return {"status": "Documents reloaded successfully"}
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))