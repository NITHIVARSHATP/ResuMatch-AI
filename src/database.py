import os
import sqlite3
import json
from src.logger import logger

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "candidates.db")
RESUMES_DIR = os.path.join(DB_DIR, "resumes")

def init_db():
    """Initializes database files and candidate tables."""
    try:
        # Create directories if they don't exist
        os.makedirs(DB_DIR, exist_ok=True)
        os.makedirs(RESUMES_DIR, exist_ok=True)
        
        # Connect to db and create candidates table
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                parsed_details TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite Database and directories initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SQLite Database: {e}")
        raise e

def save_candidate(filename: str, name: str, raw_text: str, parsed_details: dict, file_bytes: bytes):
    """
    Saves candidate details into the database and writes original file bytes to local resumes folder.
    """
    try:
        init_db()  # Ensure DB and folders are initialized
        
        # Save original file bytes to folder
        file_path = os.path.join(RESUMES_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Saved raw resume file: {file_path}")
        
        # Save info to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        parsed_details_json = json.dumps(parsed_details)
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO candidates (filename, name, raw_text, parsed_details)
            VALUES (?, ?, ?, ?)
            """,
            (filename, name, raw_text, parsed_details_json)
        )
        conn.commit()
        conn.close()
        logger.info(f"Candidate '{name}' ({filename}) successfully saved to database.")
    except Exception as e:
        logger.error(f"Error saving candidate {name} ({filename}): {e}")
        raise e

def load_candidates():
    """
    Loads all candidates from the SQLite database.
    Returns a list of dicts.
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        candidates = []
        for row in rows:
            candidate = dict(row)
            # Parse the details JSON back to python dict
            try:
                candidate["parsed_details"] = json.loads(candidate["parsed_details"])
            except json.JSONDecodeError:
                candidate["parsed_details"] = {}
            candidates.append(candidate)
            
        return candidates
    except Exception as e:
        logger.error(f"Error loading candidates: {e}")
        return []

def delete_candidate(filename: str):
    """
    Deletes candidate from database and deletes their stored resume file.
    """
    try:
        init_db()
        # Delete stored file
        file_path = os.path.join(RESUMES_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted raw resume file: {file_path}")
        else:
            logger.warning(f"Resume file to delete not found: {file_path}")
            
        # Delete from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM candidates WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
        logger.info(f"Candidate file '{filename}' removed from SQLite database.")
        return True
    except Exception as e:
        logger.error(f"Error deleting candidate '{filename}': {e}")
        return False

def is_candidate_exists(filename: str) -> bool:
    """
    Checks if a candidate resume filename already exists in the database.
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM candidates WHERE filename = ? LIMIT 1", (filename,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking if candidate exists: {e}")
        return False

def wipe_db():
    """
    Deletes all candidates from the database and wipes all saved resume files.
    """
    try:
        # Delete resume files
        if os.path.exists(RESUMES_DIR):
            for file in os.listdir(RESUMES_DIR):
                file_path = os.path.join(RESUMES_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("All stored resume files deleted.")
            
        # Reinitialize database structure
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info("SQLite DB file deleted.")
            
        init_db()
        return True
    except Exception as e:
        logger.error(f"Error wiping database: {e}")
        return False
