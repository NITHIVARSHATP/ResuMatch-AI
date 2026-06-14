import os
import io
import docx2txt
from pypdf import PdfReader
import pdfplumber
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logger import logger

# 1. Define Pydantic structured schemas
class EducationItem(BaseModel):
    degree: str = Field(description="The degree or certification obtained (e.g. B.S. in Computer Science)")
    institution: str = Field(description="The institution or university name")
    graduation_year: Optional[str] = Field(None, description="The year of graduation or completion")

class ExperienceItem(BaseModel):
    job_title: str = Field(description="The title of the position held")
    company: str = Field(description="The name of the company or organization")
    duration: Optional[str] = Field(None, description="The timeframe/dates of employment (e.g. June 2021 - Present)")
    description: Optional[str] = Field(None, description="Brief description of responsibilities and achievements")

class CandidateDetails(BaseModel):
    name: str = Field("Unknown Candidate", description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address of the candidate")
    phone: Optional[str] = Field(None, description="Phone number or contact number")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    skills: List[str] = Field(default_factory=list, description="List of technical skills, tools, and certifications")
    education: List[EducationItem] = Field(default_factory=list, description="Education history")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience details")

# 2. Document Text Extractors
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        # Try pypdf first
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        logger.warning(f"pypdf extraction failed, falling back to pdfplumber: {e}")
        text = ""

    if not text.strip():
        try:
            pdf_file = io.BytesIO(file_bytes)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise e
    return text

def extract_text_from_docx(file_bytes: bytes, filename: str) -> str:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        text = docx2txt.process(temp_path)
        return text
    except Exception as e:
        logger.error(f"docx2txt extraction failed: {e}")
        raise e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts text from file bytes depending on the file format."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes, filename)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# 3. LLM Parsing
def parse_resume_text(text: str, api_key: Optional[str] = None) -> CandidateDetails:
    """
    Uses Google Gemini model via LangChain to structure resume text into Pydantic CandidateDetails.
    """
    # Fetch active API key
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing. Please configure it in .env or via the Sidebar.")

    try:
        # Initialize Gemini Chat model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=key
        )
        
        # Configure model to yield structured output mapping to CandidateDetails schema
        structured_llm = llm.with_structured_output(CandidateDetails)
        
        prompt = (
            "You are an expert AI resume screening assistant. Analyze the following resume text "
            "and extract structured candidate information. Be careful to extract all technical skills, "
            "certifications, education, and work history accurately. If any information is missing "
            "or not explicitly present, leave the corresponding fields empty/null.\n\n"
            f"Resume Text:\n{text}"
        )
        
        logger.info("Sending resume text to Gemini for structured schema parsing...")
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error parsing resume text via Google Gemini: {e}")
        raise e
