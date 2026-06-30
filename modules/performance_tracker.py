"""
Performance Tracker Module
Analyzes user quiz performance and generates personalized feedback
and study recommendations.
"""

from typing import List, Dict, Optional
from datetime import datetime


class PerformanceTracker:
    """Tracks and analyzes user performance across quiz attempts.
    
    Provides difficulty recommendations, per-quiz feedback,
    study recommendations, and progress summaries.
    """
    
    def __init__(self, db_handler):
        """Initialize with a DatabaseHandler instance.
        
        Args:
            db_handler: An instance of DatabaseHandler for data access.
        """
        self.db = db_handler
    
    def get_recommended_difficulty(self, username: str) -> str:
        """Determine the recommended difficulty level for a user.
        
        Uses performance history to suggest an appropriate challenge level.
        
        Args:
            username: The user to analyze.
        
        Returns:
            Recommended difficulty: 'easy', 'medium', or 'hard'.
        """
        perf = self.db.get_performance_by_difficulty(username)
        
        easy = perf.get('easy', {})
        medium = perf.get('medium', {})
        hard = perf.get('hard', {})
        
        # No attempts yet - start easy
        total_attempts = easy.get('attempts', 0) + medium.get('attempts', 0) + hard.get('attempts', 0)
        if total_attempts == 0:
            return 'easy'
        
        # Check for level-up conditions
        if easy.get('avg_score', 0) >= 80 and easy.get('attempts', 0) >= 2:
            if medium.get('attempts', 0) == 0 or medium.get('avg_score', 0) >= 70:
                if medium.get('avg_score', 0) >= 70 and medium.get('attempts', 0) >= 2:
                    return 'hard'
                return 'medium'
        
        # Check for level-down conditions
        if hard.get('avg_score', 0) < 50 and hard.get('attempts', 0) >= 1:
            return 'medium'
        
        if medium.get('avg_score', 0) < 40 and medium.get('attempts', 0) >= 1:
            return 'easy'
        
        # Default: stay at the highest level attempted
        if hard.get('attempts', 0) > 0:
            return 'hard'
        if medium.get('attempts', 0) > 0:
            return 'medium'
        return 'easy'
    
    def generate_quiz_feedback(self, quiz_result: Dict) -> Dict:
        """Generate detailed feedback for a completed quiz.
        
        Args:
            quiz_result: Dict with score, total, difficulty, questions_detail.
        
        Returns:
            Feedback dict with performance label, message, and suggestions.
        """
        score = quiz_result.get('score', 0)
        total = quiz_result.get('total', 1)
        difficulty = quiz_result.get('difficulty', 'medium')
        questions_detail = quiz_result.get('questions_detail', [])
        
        percentage = (score / total * 100) if total > 0 else 0
        
        # Determine performance label and message
        if percentage >= 90:
            label = 'Excellent! 🌟'
            message = f"Outstanding performance! You scored {score}/{total} ({percentage:.0f}%) on {difficulty} level. You have a strong grasp of this material!"
        elif percentage >= 70:
            label = 'Good Job! 👍'
            message = f"Well done! You scored {score}/{total} ({percentage:.0f}%) on {difficulty} level. Keep up the great work — you're on the right track!"
        elif percentage >= 50:
            label = 'Needs Improvement 📚'
            message = f"You scored {score}/{total} ({percentage:.0f}%) on {difficulty} level. Review the topics you missed and try again. Practice makes perfect!"
        else:
            label = 'Keep Practicing 💪'
            message = f"You scored {score}/{total} ({percentage:.0f}%) on {difficulty} level. Don't be discouraged! Focus on understanding the source material and try a lower difficulty first."
        
        # Collect wrong answers
        wrong_answers = []
        for q in questions_detail:
            if not q.get('is_correct', False):
                wrong_answers.append({
                    'question': q.get('question', ''),
                    'your_answer': q.get('user_answer', ''),
                    'correct_answer': q.get('correct_answer', ''),
                    'explanation': q.get('explanation', '')
                })
        
        # Level suggestions
        should_level_up = percentage >= 80
        should_level_down = percentage < 40
        
        if should_level_up:
            next_level = {'easy': 'medium', 'medium': 'hard', 'hard': 'hard'}
            suggested = next_level.get(difficulty, difficulty)
        elif should_level_down:
            prev_level = {'easy': 'easy', 'medium': 'easy', 'hard': 'medium'}
            suggested = prev_level.get(difficulty, difficulty)
        else:
            suggested = difficulty
        
        return {
            'score_percentage': round(percentage, 1),
            'performance_label': label,
            'message': message,
            'wrong_answers': wrong_answers,
            'should_level_up': should_level_up,
            'should_level_down': should_level_down,
            'suggested_difficulty': suggested
        }
    
    def get_study_recommendations(self, username: str, chunks: List[str] = None) -> List[Dict]:
        """Generate personalized study recommendations.
        
        Args:
            username: The user to generate recommendations for.
            chunks: Optional list of document chunks for context-specific advice.
        
        Returns:
            List of recommendation dicts with recommendation, priority, reason.
        """
        recommendations = []
        perf = self.db.get_performance_by_difficulty(username)
        recent = self.db.get_recent_attempts(username, limit=5)
        
        # Recommendation 1: Based on weak difficulty levels
        weakest_level = None
        lowest_score = 100
        for level in ['easy', 'medium', 'hard']:
            level_data = perf.get(level, {})
            if level_data.get('attempts', 0) > 0 and level_data.get('avg_score', 100) < lowest_score:
                lowest_score = level_data['avg_score']
                weakest_level = level
        
        if weakest_level and lowest_score < 70:
            recommendations.append({
                'recommendation': f"Focus on improving your {weakest_level}-level performance",
                'priority': 'High',
                'reason': f"Your average score at {weakest_level} level is {lowest_score:.0f}%, which is below the target of 70%.",
                'related_chunks': []
            })
        
        # Recommendation 2: Based on recent wrong answers
        wrong_topics = []
        for attempt in recent:
            for q in attempt.get('questions_detail', []):
                if not q.get('is_correct', False):
                    wrong_topics.append(q.get('correct_answer', ''))
        
        if wrong_topics:
            topic_preview = ', '.join(wrong_topics[:5])
            related = []
            if chunks:
                for chunk in chunks[:3]:
                    for topic in wrong_topics[:3]:
                        if topic.lower() in chunk.lower():
                            related.append(chunk[:200] + '...')
                            break
            
            recommendations.append({
                'recommendation': f"Review these frequently missed topics: {topic_preview}",
                'priority': 'High',
                'reason': 'These topics appeared in your recent incorrect answers.',
                'related_chunks': related
            })
        
        # Recommendation 3: Difficulty progression advice
        recommended_diff = self.get_recommended_difficulty(username)
        recommendations.append({
            'recommendation': f"Your recommended difficulty level is: {recommended_diff.upper()}",
            'priority': 'Medium',
            'reason': f"Based on your performance history, the {recommended_diff} level provides the best balance of challenge and learning.",
            'related_chunks': []
        })
        
        # Recommendation 4: Study strategy
        total_attempts = sum(perf.get(l, {}).get('attempts', 0) for l in ['easy', 'medium', 'hard'])
        if total_attempts < 3:
            strategy = "Take at least 3 quizzes to build a performance baseline. Start with Easy level and work your way up."
        elif total_attempts < 10:
            strategy = "You're building momentum! Focus on consistency — try to complete at least one quiz per study session."
        else:
            strategy = "Great dedication! Focus on your weak areas and try to maintain above 70% at each difficulty level before moving up."
        
        recommendations.append({
            'recommendation': strategy,
            'priority': 'Low',
            'reason': f"Based on your {total_attempts} total quiz attempts.",
            'related_chunks': []
        })
        
        return recommendations
    
    def get_progress_summary(self, username: str) -> Dict:
        """Get an overall progress summary for the user.
        
        Args:
            username: The user to summarize.
        
        Returns:
            Dict with total stats, strongest/weakest levels, trend, and streak.
        """
        perf = self.db.get_performance_by_difficulty(username)
        attempts = self.db.get_user_attempts(username)
        
        total_quizzes = len(attempts)
        total_questions = sum(a.get('total', 0) for a in attempts)
        total_correct = sum(a.get('score', 0) for a in attempts)
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Find strongest and weakest levels
        strongest = 'N/A'
        weakest = 'N/A'
        best_score = -1
        worst_score = 101
        
        for level in ['easy', 'medium', 'hard']:
            level_data = perf.get(level, {})
            if level_data.get('attempts', 0) > 0:
                avg = level_data['avg_score']
                if avg > best_score:
                    best_score = avg
                    strongest = level
                if avg < worst_score:
                    worst_score = avg
                    weakest = level
        
        # Calculate improvement trend
        trend = 'Stable'
        if len(attempts) >= 6:
            recent_3 = attempts[:3]
            previous_3 = attempts[3:6]
            recent_avg = sum(a.get('score_percentage', 0) for a in recent_3) / 3
            prev_avg = sum(a.get('score_percentage', 0) for a in previous_3) / 3
            if recent_avg > prev_avg + 5:
                trend = 'Improving 📈'
            elif recent_avg < prev_avg - 5:
                trend = 'Declining 📉'
            else:
                trend = 'Stable ➡️'
        elif len(attempts) >= 2:
            if attempts[0].get('score_percentage', 0) > attempts[1].get('score_percentage', 0):
                trend = 'Improving 📈'
            else:
                trend = 'Stable ➡️'
        
        # Calculate streak (consecutive quizzes with >= 60%)
        streak = 0
        for attempt in attempts:
            if attempt.get('score_percentage', 0) >= 60:
                streak += 1
            else:
                break
        
        return {
            'total_quizzes': total_quizzes,
            'total_questions_answered': total_questions,
            'overall_accuracy': round(overall_accuracy, 1),
            'strongest_level': strongest,
            'weakest_level': weakest,
            'improvement_trend': trend,
            'streak': streak
        }
