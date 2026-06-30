"""
Database Handler Module
Manages data persistence with MongoDB Atlas (primary) and local JSON file (fallback).
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class DatabaseHandler:
    """Handles all database operations with dual-mode storage.
    
    Attempts to connect to MongoDB Atlas first. If unavailable,
    falls back to local JSON file storage for development/demo use.
    """
    
    def __init__(self):
        """Initialize database connection.
        
        Tries MongoDB Atlas first (via Streamlit secrets or env var),
        then falls back to local JSON storage.
        """
        self.use_mongo = False
        self.db = None
        self.json_path = 'user_data.json'
        
        # Try MongoDB connection
        try:
            mongo_uri = None
            
            # Method 1: Streamlit secrets
            try:
                import streamlit as st
                mongo_uri = st.secrets.get('MONGODB_URI', None)
            except Exception:
                pass
            
            # Method 2: Environment variable
            if not mongo_uri:
                mongo_uri = os.environ.get('MONGODB_URI', None)
            
            if mongo_uri:
                from pymongo import MongoClient
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                # Test connection
                client.server_info()
                self.db = client['rag_mcq_system']
                self.use_mongo = True
                print("✅ Connected to MongoDB Atlas")
            else:
                print("ℹ️ No MongoDB URI found. Using local JSON storage.")
        except Exception as e:
            print(f"⚠️ MongoDB connection failed: {e}. Using local JSON storage.")
        
        # Initialize JSON storage if needed
        if not self.use_mongo:
            self._init_json()
    
    def _init_json(self):
        """Initialize the JSON file with empty structure if it doesn't exist."""
        if not os.path.exists(self.json_path):
            self._save_json({
                'users': [],
                'quiz_attempts': [],
                'performance_summary': []
            })
    
    def _load_json(self) -> Dict:
        """Load data from the JSON file.
        
        Returns:
            Dictionary containing all stored data.
        """
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            default = {'users': [], 'quiz_attempts': [], 'performance_summary': []}
            self._save_json(default)
            return default
    
    def _save_json(self, data: Dict):
        """Save data to the JSON file.
        
        Args:
            data: Dictionary containing all data to persist.
        """
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_user(self, username: str) -> bool:
        """Create a new user record.
        
        Args:
            username: The username to create.
        
        Returns:
            True if user was created, False if already exists.
        """
        if self.user_exists(username):
            return False
        
        user_record = {
            'username': username,
            'created_at': datetime.now().isoformat()
        }
        
        if self.use_mongo:
            self.db['users'].insert_one(user_record)
        else:
            data = self._load_json()
            data['users'].append(user_record)
            self._save_json(data)
        
        return True
    
    def user_exists(self, username: str) -> bool:
        """Check if a username already exists.
        
        Args:
            username: The username to check.
        
        Returns:
            True if the user exists.
        """
        if self.use_mongo:
            return self.db['users'].find_one({'username': username}) is not None
        else:
            data = self._load_json()
            return any(u['username'] == username for u in data['users'])
    
    def save_quiz_attempt(self, username: str, pdf_name: str, difficulty: str,
                          score: int, total: int, questions_detail: List[Dict]) -> None:
        """Save a quiz attempt record.
        
        Args:
            username: The user who took the quiz.
            pdf_name: Name of the PDF document.
            difficulty: Quiz difficulty level.
            score: Number of correct answers.
            total: Total number of questions.
            questions_detail: List of question details with user answers.
        """
        attempt = {
            'username': username,
            'pdf_name': pdf_name,
            'difficulty': difficulty,
            'score': score,
            'total': total,
            'score_percentage': round((score / total * 100) if total > 0 else 0, 1),
            'questions_detail': questions_detail,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.use_mongo:
            self.db['quiz_attempts'].insert_one(attempt)
        else:
            data = self._load_json()
            data['quiz_attempts'].append(attempt)
            self._save_json(data)
    
    def get_user_attempts(self, username: str) -> List[Dict]:
        """Get all quiz attempts for a user, sorted by timestamp descending.
        
        Args:
            username: The username to query.
        
        Returns:
            List of attempt records, newest first.
        """
        if self.use_mongo:
            attempts = list(self.db['quiz_attempts'].find(
                {'username': username},
                {'_id': 0}
            ).sort('timestamp', -1))
        else:
            data = self._load_json()
            attempts = [a for a in data['quiz_attempts'] if a['username'] == username]
            attempts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return attempts
    
    def get_performance_by_difficulty(self, username: str) -> Dict:
        """Aggregate performance statistics by difficulty level.
        
        Args:
            username: The username to analyze.
        
        Returns:
            Dict with 'easy', 'medium', 'hard' keys, each containing
            attempts, avg_score, total_questions, correct_answers.
        """
        attempts = self.get_user_attempts(username)
        
        performance = {}
        for level in ['easy', 'medium', 'hard']:
            level_attempts = [a for a in attempts if a['difficulty'] == level]
            
            if level_attempts:
                total_q = sum(a['total'] for a in level_attempts)
                correct = sum(a['score'] for a in level_attempts)
                avg_pct = sum(a.get('score_percentage', 0) for a in level_attempts) / len(level_attempts)
                
                performance[level] = {
                    'attempts': len(level_attempts),
                    'avg_score': round(avg_pct, 1),
                    'total_questions': total_q,
                    'correct_answers': correct
                }
            else:
                performance[level] = {
                    'attempts': 0,
                    'avg_score': 0.0,
                    'total_questions': 0,
                    'correct_answers': 0
                }
        
        return performance
    
    def get_recent_attempts(self, username: str, limit: int = 10) -> List[Dict]:
        """Get the most recent quiz attempts for a user.
        
        Args:
            username: The username to query.
            limit: Maximum number of attempts to return.
        
        Returns:
            List of recent attempt records.
        """
        attempts = self.get_user_attempts(username)
        return attempts[:limit]
    
    def get_storage_mode(self) -> str:
        """Get the current storage mode description.
        
        Returns:
            'MongoDB Atlas' or 'Local JSON Storage'
        """
        return 'MongoDB Atlas' if self.use_mongo else 'Local JSON Storage'
