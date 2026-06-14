import os
import unittest
import sqlite3
import shutil
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

# Adjust paths to import src modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import database
from src import parser
from src import matcher
from src import rag_engine

class TestResuMatchAI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Override database paths for isolated testing
        cls.test_db_dir = "test_data"
        cls.test_db_path = os.path.join(cls.test_db_dir, "test_candidates.db")
        cls.test_resumes_dir = os.path.join(cls.test_db_dir, "resumes")
        
        # Patch database module constants
        database.DB_DIR = cls.test_db_dir
        database.DB_PATH = cls.test_db_path
        database.RESUMES_DIR = cls.test_resumes_dir
        
    def setUp(self):
        # Clean test directory before each test
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        database.init_db()

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)

    # 1. Database Operations Tests
    def test_database_init(self):
        self.assertTrue(os.path.exists(self.test_db_path))
        self.assertTrue(os.path.exists(self.test_resumes_dir))
        
        # Check if table candidates exists
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'")
        table_exists = cursor.fetchone() is not None
        conn.close()
        self.assertTrue(table_exists)

    def test_database_save_and_load(self):
        filename = "test_resume.txt"
        name = "Jane Doe"
        raw_text = "Experienced software developer skilled in Python and Django."
        parsed_details = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "123-456-7890",
            "linkedin": "linkedin.com/in/janedoe",
            "skills": ["Python", "Django", "SQL"],
            "education": [],
            "experience": []
        }
        file_bytes = b"Jane Doe Resume Contents"
        
        # Save candidate
        database.save_candidate(filename, name, raw_text, parsed_details, file_bytes)
        
        # Verify file saved
        saved_file_path = os.path.join(self.test_resumes_dir, filename)
        self.assertTrue(os.path.exists(saved_file_path))
        with open(saved_file_path, "rb") as f:
            self.assertEqual(f.read(), file_bytes)
            
        # Verify exists
        self.assertTrue(database.is_candidate_exists(filename))
        
        # Verify loading candidates
        candidates = database.load_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["name"], name)
        self.assertEqual(candidates[0]["filename"], filename)
        self.assertEqual(candidates[0]["raw_text"], raw_text)
        self.assertEqual(candidates[0]["parsed_details"]["email"], "jane@example.com")
        
    def test_database_delete(self):
        filename = "jane_doe.txt"
        database.save_candidate(filename, "Jane Doe", "some text", {}, b"bytes")
        self.assertTrue(database.is_candidate_exists(filename))
        
        # Delete
        success = database.delete_candidate(filename)
        self.assertTrue(success)
        self.assertFalse(database.is_candidate_exists(filename))
        self.assertFalse(os.path.exists(os.path.join(self.test_resumes_dir, filename)))

    # 2. Parsing Extraction Tests
    def test_txt_extraction(self):
        text_bytes = b"Hello, this is a plain text file."
        extracted = parser.extract_text(text_bytes, "test.txt")
        self.assertEqual(extracted, "Hello, this is a plain text file.")

    def test_docx_extraction_cleanup(self):
        # We check docx2txt temp file cleanup upon failure
        # By passing invalid bytes that crash docx2txt, we make sure temp files are still cleaned up
        with self.assertRaises(Exception):
            parser.extract_text(b"invalid zip docx format", "test.docx")
            
        # Ensure temp file does not remain
        self.assertFalse(os.path.exists(os.path.join("temp", "test.docx")))

    # 3. Vector store indexing RAG Tests
    @patch('src.rag_engine.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.FAISS')
    def test_vector_store_building(self, mock_faiss, mock_embeddings):
        # Setup mock embeddings and FAISS
        mock_emb_instance = MagicMock()
        mock_embeddings.return_value = mock_emb_instance
        
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance
        
        candidates = [
            {
                "name": "Alice Smith",
                "filename": "alice.txt",
                "raw_text": "Alice is a Data Scientist with 5 years experience in machine learning."
            }
        ]
        
        vs = rag_engine.build_vector_store(candidates, api_key="dummy_api_key")
        
        self.assertIsNotNone(vs)
        mock_faiss.from_documents.assert_called_once()
        
        # Verify the structure of documents passed to FAISS
        called_args = mock_faiss.from_documents.call_args[0]
        documents = called_args[0]
        
        self.assertGreater(len(documents), 0)
        self.assertEqual(documents[0].metadata["candidate_name"], "Alice Smith")
        self.assertIn("Candidate: Alice Smith", documents[0].page_content)

    # 4. LLM Matcher & Parser Integrations (Mocked to avoid network dependancy during offline tests)
    @patch('src.parser.ChatGoogleGenerativeAI')
    def test_mock_resume_parsing(self, mock_chat):
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        
        mock_structured_output = MagicMock()
        # Mock what Gemini returns
        mock_candidate = parser.CandidateDetails(
            name="Bob Johnson",
            email="bob@example.com",
            phone="999-999-9999",
            linkedin="linkedin.com/in/bob",
            skills=["Java", "Spring Boot"],
            education=[],
            experience=[]
        )
        mock_structured_output.invoke.return_value = mock_candidate
        mock_llm.with_structured_output.return_value = mock_structured_output
        
        result = parser.parse_resume_text("Bob Johnson Java Developer resume text...", api_key="dummy_api_key")
        
        self.assertEqual(result.name, "Bob Johnson")
        self.assertEqual(result.email, "bob@example.com")
        self.assertIn("Java", result.skills)

    @patch('src.matcher.ChatGoogleGenerativeAI')
    def test_mock_match_analysis(self, mock_chat):
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        
        mock_structured_output = MagicMock()
        mock_match = matcher.MatchAnalysis(
            candidate_name="Alice Smith",
            match_percentage=85,
            key_alignment_points=["Python expertise", "ML models design"],
            missing_requirements=["Kubernetes"],
            overall_fit_explanation="Excellent technical alignment for Data Science."
        )
        mock_structured_output.invoke.return_value = mock_match
        mock_llm.with_structured_output.return_value = mock_structured_output
        
        result = matcher.analyze_candidate_match(
            resume_text="Alice Smith CV...",
            jd_text="Looking for a Python ML Engineer with Kubernetes experience.",
            candidate_name="Alice Smith",
            api_key="dummy_api_key"
        )
        
        self.assertEqual(result.candidate_name, "Alice Smith")
        self.assertEqual(result.match_percentage, 85)
        self.assertIn("Kubernetes", result.missing_requirements)

if __name__ == '__main__':
    unittest.main()
