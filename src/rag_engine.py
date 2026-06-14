import os
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.logger import logger

def build_vector_store(candidates: List[Dict[str, Any]], api_key: Optional[str] = None) -> Optional[FAISS]:
    """
    Chunks candidate resumes, embeds them, and creates a FAISS-CPU vector store.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing. Please configure it in .env or via the Sidebar.")

    if not candidates:
        logger.warning("No candidates found in database to build vector store.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []

    for candidate in candidates:
        name = candidate.get("name", "Unknown Candidate")
        raw_text = candidate.get("raw_text", "")
        filename = candidate.get("filename", "")
        
        if not raw_text.strip():
            continue
            
        chunks = text_splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            # Prefix each chunk with metadata to prevent confusion in global RAG
            prefixed_text = f"Candidate: {name}\nResume Content (Part {i+1}):\n{chunk}"
            metadata = {
                "candidate_name": name,
                "filename": filename
            }
            docs.append(Document(page_content=prefixed_text, metadata=metadata))

    if not docs:
        logger.warning("No valid text extracted to chunk.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=key
        )

        logger.info(f"Generating FAISS vector store database from {len(docs)} chunks...")
        vector_store = FAISS.from_documents(docs, embeddings)
        logger.info("FAISS vector store successfully created.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise e

def answer_candidate_query(
    vector_store: FAISS,
    query: str,
    candidate_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Queries the vector store for context and calls Gemini to synthesize an answer.
    Supports candidate filtering.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing. Please configure it in .env or via the Sidebar.")

    try:
        # Retrieve context from vector store
        if candidate_name and candidate_name.strip() and candidate_name != "All Candidates":
            logger.info(f"Performing filtered RAG search for candidate '{candidate_name}'")
            # FAISS supports dict filtering on metadata keys
            docs = vector_store.similarity_search(
                query, 
                k=10, 
                filter={"candidate_name": candidate_name}
            )
            # Pick top 5 after filter
            docs = docs[:5]
        else:
            logger.info("Performing global RAG search across all candidates")
            docs = vector_store.similarity_search(query, k=5)

        if not docs:
            return "No matching context found in the candidate resumes to answer this query."

        # Compile context
        context_parts = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "Unknown file")
            context_parts.append(f"[Source: {filename}]\n{doc.page_content}")
            
        context = "\n\n---\n\n".join(context_parts)

        # Build Synthesis Chat Prompt
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=key
        )

        qa_system_instruction = (
            "You are an expert recruiting coordinator assistant at ResuMatch AI.\n"
            "Your task is to answer the user's natural language query strictly utilizing the candidate details provided in the Context section below.\n\n"
            "Guidelines:\n"
            "1. Answer the question comprehensively and clearly.\n"
            "2. State only facts directly mentioned in the Context. Do NOT invent, assume, or extrapolate details (e.g. do not guess contact information, graduation years, or project descriptions if they aren't explicitly mentioned).\n"
            "3. If the context does not contain enough information to answer, reply explaining that the details are not available in the candidate's resume.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query:\n{query}"
        )

        logger.info("Sending query and retrieved chunks to Gemini for synthesis...")
        response = llm.invoke(qa_system_instruction)
        return response.content
    except Exception as e:
        logger.error(f"Error executing Q&A retrieval or LLM synthesis: {e}")
        return f"Error executing Q&A Assistant: {str(e)}"
