import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logger import logger

# 1. Define Pydantic structured schema for match evaluation
class MatchAnalysis(BaseModel):
    candidate_name: str = Field(description="Name of the candidate evaluated")
    match_percentage: int = Field(description="Match percentage between resume and JD (integer between 0 and 100)")
    key_alignment_points: List[str] = Field(default_factory=list, description="Strengths and alignment points of the candidate for the JD")
    missing_requirements: List[str] = Field(default_factory=list, description="Gaps, missing skills, or requirements not met by the candidate")
    overall_fit_explanation: str = Field(description="Detailed explanation of the candidate's overall fit and matching score reasons")

# 2. Match Analysis function
def analyze_candidate_match(resume_text: str, jd_text: str, candidate_name: str, api_key: Optional[str] = None) -> MatchAnalysis:
    """
    Analyzes resume content against a job description text and returns a MatchAnalysis schema.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini API Key is missing. Please configure it in .env or via the Sidebar.")

    try:
        # Initialize Gemini model at temperature 0.1
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=key
        )
        
        # Configure model to yield structured output mapping to MatchAnalysis schema
        structured_llm = llm.with_structured_output(MatchAnalysis)
        
        system_prompt = (
            "You are a Senior Recruiter and Applicant Tracking System. Your job is to objectively analyze "
            "the provided Resume Text against the Job Description (JD) and produce a detailed match assessment.\n\n"
            f"Candidate Name: {candidate_name}\n\n"
            f"Job Description (JD):\n{jd_text}\n\n"
            f"Resume Text:\n{resume_text}"
        )
        
        logger.info(f"Analyzing match fit for candidate '{candidate_name}'...")
        result = structured_llm.invoke(system_prompt)
        return result
    except Exception as e:
        logger.error(f"Error matching candidate '{candidate_name}': {e}")
        raise e
