"""
RAG-Powered Adaptive MCQ Generation System
Main Streamlit Application

A B.Tech final year project that generates adaptive MCQs from PDF documents
using Retrieval-Augmented Generation (RAG) with TF-IDF and cosine similarity.
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px

from modules.pdf_processor import extract_text_from_pdf, chunk_text, get_text_stats
from modules.retriever import TFIDFRetriever, extract_keywords
from modules.mcq_generator import generate_mcqs
from modules.db_handler import DatabaseHandler
from modules.performance_tracker import PerformanceTracker

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG MCQ Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for Premium Dark UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1D29 50%, #0E1117 100%);
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #8892B0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1E2235 0%, #252A3A 100%);
        border: 1px solid #2D3348;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8892B0;
        margin-top: 0.25rem;
    }
    
    /* Quiz option cards */
    .quiz-option {
        background: #1E2235;
        border: 1px solid #2D3348;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Correct / Incorrect feedback */
    .correct-answer {
        background: linear-gradient(135deg, #1a3a2a 0%, #1E2235 100%);
        border-left: 4px solid #4ECDC4;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .wrong-answer {
        background: linear-gradient(135deg, #3a1a1a 0%, #1E2235 100%);
        border-left: 4px solid #FF6B6B;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Difficulty badges */
    .badge-easy {
        background: #4ECDC4; color: #0E1117;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }
    .badge-medium {
        background: #FFD93D; color: #0E1117;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }
    .badge-hard {
        background: #FF6B6B; color: #0E1117;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; display: inline-block;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141824 0%, #1A1D29 100%);
        border-right: 1px solid #2D3348;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF 0%, #6C63FF 100%);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
        transform: translateY(-1px);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1a2040 0%, #1E2235 100%);
        border: 1px solid #6C63FF40;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Recommendation card */
    .rec-card {
        background: #1E2235;
        border-left: 4px solid #6C63FF;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
    }
    .rec-high { border-left-color: #FF6B6B; }
    .rec-medium { border-left-color: #FFD93D; }
    .rec-low { border-left-color: #4ECDC4; }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Initialize Session State
# ─────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'page': '📄 Upload PDF',
        'username': '',
        'logged_in': False,
        'pdf_text': '',
        'pdf_name': '',
        'chunks': [],
        'retriever': None,
        'text_stats': {},
        'quiz_questions': [],
        'quiz_submitted': False,
        'quiz_answers': {},
        'quiz_difficulty': 'easy',
        'quiz_feedback': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─────────────────────────────────────────────────────────────
# Initialize Database & Performance Tracker
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_db_handler():
    """Get or create the database handler (cached)."""
    return DatabaseHandler()

@st.cache_resource
def get_performance_tracker(_db):
    """Get or create the performance tracker (cached)."""
    return PerformanceTracker(_db)

db = get_db_handler()
tracker = get_performance_tracker(db)


# ─────────────────────────────────────────────────────────────
# Sidebar — Navigation & User Login
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.6rem;">🧠 RAG MCQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="font-size:0.85rem;">Adaptive Quiz Generator</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # User login / registration
    st.markdown("### 👤 User Profile")
    
    if not st.session_state.logged_in:
        username_input = st.text_input("Enter your username", placeholder="e.g., student_01")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                if username_input:
                    if db.user_exists(username_input):
                        st.session_state.username = username_input
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("User not found!")
        with col2:
            if st.button("Register", use_container_width=True):
                if username_input:
                    if db.create_user(username_input):
                        st.session_state.username = username_input
                        st.session_state.logged_in = True
                        st.success("Registered! ✅")
                        st.rerun()
                    else:
                        st.warning("Username taken!")
    else:
        st.success(f"Welcome, **{st.session_state.username}**!")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.rerun()
    
    st.divider()
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    pages = ['📄 Upload PDF', '📝 Take Quiz', '📊 Dashboard', '💡 Recommendations']
    
    for page in pages:
        if st.button(page, use_container_width=True, 
                     type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()
    
    st.divider()
    
    # Status indicators
    st.markdown("### ⚙️ System Status")
    st.markdown(f"**Storage:** {db.get_storage_mode()}")
    if st.session_state.chunks:
        st.markdown(f"**PDF Loaded:** ✅ {st.session_state.pdf_name}")
        st.markdown(f"**Chunks:** {len(st.session_state.chunks)}")
    else:
        st.markdown("**PDF Loaded:** ❌ None")


# ─────────────────────────────────────────────────────────────
# Page: Upload PDF
# ─────────────────────────────────────────────────────────────
def page_upload():
    st.markdown('<h1 class="main-header">📄 Upload Your PDF Document</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a textbook, article, or lecture notes to generate quiz questions</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload any PDF document. The system will extract text and create intelligent quiz questions."
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing your PDF..."):
            # Extract text
            progress = st.progress(0, text="Extracting text from PDF...")
            text = extract_text_from_pdf(uploaded_file)
            progress.progress(33, text="Chunking text into passages...")
            
            if not text:
                st.error("❌ Could not extract text from this PDF. Please try another file.")
                return
            
            # Chunk text
            chunks = chunk_text(text, chunk_size=5, overlap=2)
            progress.progress(66, text="Building search index...")
            
            if not chunks:
                st.error("❌ Not enough text content found in the PDF.")
                return
            
            # Build retriever index
            retriever = TFIDFRetriever()
            retriever.fit(chunks)
            progress.progress(100, text="✅ Processing complete!")
            time.sleep(0.5)
            
            # Save to session state
            st.session_state.pdf_text = text
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.text_stats = get_text_stats(text, chunks)
        
        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
        
        # Display statistics
        stats = st.session_state.text_stats
        st.markdown("### 📊 Document Statistics")
        
        cols = st.columns(5)
        stat_items = [
            ("📝", "Characters", f"{stats['total_characters']:,}"),
            ("📖", "Words", f"{stats['total_words']:,}"),
            ("📋", "Sentences", f"{stats['total_sentences']:,}"),
            ("🧩", "Chunks", f"{stats['total_chunks']}"),
            ("📏", "Avg Chunk", f"{stats['avg_chunk_length']:.0f} chars"),
        ]
        for col, (icon, label, value) in zip(cols, stat_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show keywords
        st.markdown("### 🔑 Top Keywords Extracted")
        keywords = extract_keywords(text, top_n=15)
        if keywords:
            keyword_html = " ".join(
                f'<span style="background:#6C63FF30;color:#A5A0FF;padding:0.3rem 0.8rem;'
                f'border-radius:20px;margin:0.2rem;display:inline-block;font-size:0.9rem;">{kw}</span>'
                for kw in keywords
            )
            st.markdown(keyword_html, unsafe_allow_html=True)
        
        # Preview chunks
        with st.expander("👁️ Preview Text Chunks (first 3)"):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(f'<div class="info-box">{chunk[:500]}{"..." if len(chunk) > 500 else ""}</div>', unsafe_allow_html=True)
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        <div class="info-box">
            <h3>📌 How It Works</h3>
            <ol>
                <li><strong>Upload</strong> a PDF document (textbook, article, notes)</li>
                <li><strong>The system extracts</strong> text and splits it into meaningful chunks</li>
                <li><strong>TF-IDF indexing</strong> enables fast retrieval of relevant passages</li>
                <li><strong>Take a quiz</strong> with adaptive difficulty (Easy → Medium → Hard)</li>
                <li><strong>Track your progress</strong> and get personalized study recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Page: Take Quiz
# ─────────────────────────────────────────────────────────────
def page_quiz():
    st.markdown('<h1 class="main-header">📝 Take a Quiz</h1>', unsafe_allow_html=True)
    
    # Pre-checks
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login from the sidebar first.")
        return
    
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload a PDF document first.")
        return
    
    retriever = st.session_state.retriever
    
    # Quiz setup (only show if no active quiz)
    if not st.session_state.quiz_questions:
        st.markdown('<p class="sub-header">Configure your quiz settings and start learning!</p>', unsafe_allow_html=True)
        
        # Get recommended difficulty
        recommended = tracker.get_recommended_difficulty(st.session_state.username)
        
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox(
                "Select Difficulty Level",
                ['easy', 'medium', 'hard'],
                index=['easy', 'medium', 'hard'].index(recommended),
                format_func=lambda x: f"{'🟢 Easy' if x == 'easy' else '🟡 Medium' if x == 'medium' else '🔴 Hard'}"
            )
            badge_class = f"badge-{difficulty}"
            st.markdown(f'<span class="{badge_class}">Recommended: {recommended.upper()}</span>', unsafe_allow_html=True)
        
        with col2:
            num_questions = st.slider("Number of Questions", min_value=3, max_value=10, value=5)
        
        # Topic input for targeted retrieval
        topic = st.text_input(
            "📌 Enter a topic or keyword (optional)",
            placeholder="e.g., machine learning, photosynthesis, civil rights...",
            help="Leave blank to generate questions from the entire document."
        )
        
        if st.button("🚀 Generate Quiz", use_container_width=True, type="primary"):
            with st.spinner("🔄 Generating questions..."):
                # Use topic for retrieval, or use keywords from the document
                query = topic if topic.strip() else " ".join(extract_keywords(st.session_state.pdf_text, top_n=10))
                
                # Retrieve relevant chunks
                relevant_chunks = retriever.retrieve_for_difficulty(query, difficulty, num_chunks=5)
                
                if not relevant_chunks:
                    st.error("❌ Could not find relevant content. Try a different topic.")
                    return
                
                # Generate MCQs
                mcqs = generate_mcqs(relevant_chunks, difficulty=difficulty, num_questions=num_questions)
                
                if not mcqs:
                    st.error("❌ Could not generate enough questions. Try a different topic or difficulty.")
                    return
                
                st.session_state.quiz_questions = mcqs
                st.session_state.quiz_difficulty = difficulty
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.quiz_feedback = None
                st.rerun()
    
    # Display quiz questions
    if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
        questions = st.session_state.quiz_questions
        difficulty = st.session_state.quiz_difficulty
        
        badge_class = f"badge-{difficulty}"
        st.markdown(f"""
        <div class="info-box">
            <span class="{badge_class}">{difficulty.upper()}</span>
            &nbsp;&nbsp;📋 <strong>{len(questions)} Questions</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;📄 From: <strong>{st.session_state.pdf_name}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Render each question
        for i, mcq in enumerate(questions):
            st.markdown(f"---")
            st.markdown(f"### Question {i + 1} of {len(questions)}")
            st.markdown(f"**{mcq['question']}**")
            
            option_labels = ['A', 'B', 'C', 'D']
            options_display = [f"{label}. {opt}" for label, opt in zip(option_labels, mcq['options'])]
            
            answer = st.radio(
                f"Select your answer for Q{i+1}:",
                options=options_display,
                key=f"q_{i}",
                index=None,
                label_visibility="collapsed"
            )
            
            if answer:
                selected_index = options_display.index(answer)
                st.session_state.quiz_answers[i] = selected_index
        
        st.markdown("---")
        
        # Submit button
        answered = len(st.session_state.quiz_answers)
        total = len(questions)
        st.progress(answered / total, text=f"Answered: {answered}/{total}")
        
        if st.button(f"✅ Submit Quiz ({answered}/{total} answered)", 
                     use_container_width=True, type="primary",
                     disabled=(answered < total)):
            st.session_state.quiz_submitted = True
            st.rerun()
    
    # Show results after submission
    if st.session_state.quiz_submitted and st.session_state.quiz_questions:
        _show_quiz_results()


def _show_quiz_results():
    """Display quiz results with detailed feedback."""
    questions = st.session_state.quiz_questions
    answers = st.session_state.quiz_answers
    difficulty = st.session_state.quiz_difficulty
    
    # Calculate score
    score = 0
    questions_detail = []
    
    for i, mcq in enumerate(questions):
        user_idx = answers.get(i, -1)
        is_correct = (user_idx == mcq['correct_index'])
        if is_correct:
            score += 1
        
        questions_detail.append({
            'question': mcq['question'],
            'user_answer': mcq['options'][user_idx] if user_idx >= 0 else 'Not answered',
            'correct_answer': mcq['correct_answer'],
            'is_correct': is_correct,
            'explanation': mcq.get('explanation', '')
        })
    
    total = len(questions)
    
    # Save to database
    db.save_quiz_attempt(
        username=st.session_state.username,
        pdf_name=st.session_state.pdf_name,
        difficulty=difficulty,
        score=score,
        total=total,
        questions_detail=questions_detail
    )
    
    # Generate feedback
    quiz_result = {
        'score': score, 'total': total,
        'difficulty': difficulty, 'questions_detail': questions_detail
    }
    feedback = tracker.generate_quiz_feedback(quiz_result)
    
    # Display score banner
    percentage = feedback['score_percentage']
    color = '#4ECDC4' if percentage >= 70 else '#FFD93D' if percentage >= 50 else '#FF6B6B'
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #1E2235, #252A3A);border:2px solid {color};
                border-radius:20px;padding:2rem;text-align:center;margin:1rem 0;">
        <div style="font-size:3rem;font-weight:700;color:{color};">{score}/{total}</div>
        <div style="font-size:1.5rem;margin-top:0.5rem;">{feedback['performance_label']}</div>
        <div style="color:#8892B0;margin-top:0.5rem;">{feedback['message']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Difficulty suggestion
    if feedback['should_level_up'] and difficulty != 'hard':
        st.success(f"🎯 Great job! You're ready to try **{feedback['suggested_difficulty'].upper()}** difficulty!")
    elif feedback['should_level_down'] and difficulty != 'easy':
        st.info(f"💡 Consider trying **{feedback['suggested_difficulty'].upper()}** difficulty to build your foundations.")
    
    # Detailed question review
    st.markdown("### 📋 Detailed Review")
    for i, (mcq, detail) in enumerate(zip(questions, questions_detail)):
        css_class = "correct-answer" if detail['is_correct'] else "wrong-answer"
        icon = "✅" if detail['is_correct'] else "❌"
        
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{icon} Q{i+1}: {mcq['question']}</strong><br/>
            <span>Your answer: <strong>{detail['user_answer']}</strong></span><br/>
            <span>Correct answer: <strong>{detail['correct_answer']}</strong></span><br/>
            <small style="color:#8892B0;">{detail['explanation']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Another Quiz", use_container_width=True, type="primary"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.rerun()
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = '📊 Dashboard'
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Page: Performance Dashboard
# ─────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown('<h1 class="main-header">📊 Performance Dashboard</h1>', unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login from the sidebar first.")
        return
    
    username = st.session_state.username
    progress = tracker.get_progress_summary(username)
    perf_by_diff = db.get_performance_by_difficulty(username)
    
    if progress['total_quizzes'] == 0:
        st.info("📝 No quiz attempts yet. Take a quiz to see your performance here!")
        return
    
    # Top-level metrics
    st.markdown("### 🏆 Overall Performance")
    cols = st.columns(4)
    metrics = [
        ("📝", "Quizzes Taken", str(progress['total_quizzes'])),
        ("🎯", "Overall Accuracy", f"{progress['overall_accuracy']:.1f}%"),
        ("🔥", "Current Streak", str(progress['streak'])),
        ("📈", "Trend", progress['improvement_trend']),
    ]
    for col, (icon, label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance by difficulty — bar chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Score by Difficulty Level")
        
        levels = ['Easy', 'Medium', 'Hard']
        scores = [perf_by_diff[l.lower()]['avg_score'] for l in levels]
        attempts_list = [perf_by_diff[l.lower()]['attempts'] for l in levels]
        colors = ['#4ECDC4', '#FFD93D', '#FF6B6B']
        
        fig = go.Figure(data=[
            go.Bar(
                x=levels,
                y=scores,
                marker_color=colors,
                text=[f"{s:.0f}%" for s in scores],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Score: %{y:.1f}%<br>Attempts: %{customdata}<extra></extra>',
                customdata=attempts_list
            )
        ])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA',
            yaxis=dict(range=[0, 110], title='Average Score (%)'),
            xaxis=dict(title=''),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Attempts Distribution")
        
        fig2 = go.Figure(data=[
            go.Pie(
                labels=levels,
                values=attempts_list,
                marker_colors=colors,
                hole=0.5,
                textinfo='label+value',
                textposition='outside'
            )
        ])
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recent quiz history — trend line
    st.markdown("### 📜 Recent Quiz History")
    recent = db.get_recent_attempts(username, limit=15)
    
    if recent:
        # Score trend chart
        timestamps = [a.get('timestamp', '')[:10] for a in reversed(recent)]
        score_pcts = [a.get('score_percentage', 0) for a in reversed(recent)]
        difficulties = [a.get('difficulty', 'unknown') for a in reversed(recent)]
        diff_colors = {'easy': '#4ECDC4', 'medium': '#FFD93D', 'hard': '#FF6B6B'}
        marker_colors = [diff_colors.get(d, '#6C63FF') for d in difficulties]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(range(1, len(score_pcts) + 1)),
            y=score_pcts,
            mode='lines+markers',
            line=dict(color='#6C63FF', width=3),
            marker=dict(size=10, color=marker_colors, line=dict(width=2, color='#FAFAFA')),
            hovertemplate='Quiz #%{x}<br>Score: %{y:.0f}%<br>Level: %{customdata}<extra></extra>',
            customdata=difficulties
        ))
        # Target line at 70%
        fig3.add_hline(y=70, line_dash="dash", line_color="#4ECDC440",
                       annotation_text="Target (70%)", annotation_font_color="#4ECDC4")
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA',
            xaxis=dict(title='Quiz Number'),
            yaxis=dict(title='Score (%)', range=[0, 105]),
            height=300,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Recent attempts table
        with st.expander("📋 Detailed Attempt History"):
            for i, attempt in enumerate(recent[:10]):
                diff = attempt.get('difficulty', 'unknown')
                badge = f"badge-{diff}"
                st.markdown(f"""
                <div class="quiz-option">
                    <span class="{badge}">{diff.upper()}</span>
                    &nbsp;&nbsp;<strong>{attempt.get('score', 0)}/{attempt.get('total', 0)}</strong>
                    ({attempt.get('score_percentage', 0):.0f}%)
                    &nbsp;&nbsp;|&nbsp;&nbsp;📄 {attempt.get('pdf_name', 'Unknown')}
                    &nbsp;&nbsp;|&nbsp;&nbsp;🕐 {attempt.get('timestamp', '')[:16]}
                </div>
                """, unsafe_allow_html=True)
    
    # Strongest / Weakest
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.2rem;">💪 Strongest Level</div>
            <div class="metric-value">{progress['strongest_level'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.2rem;">📚 Needs Work</div>
            <div class="metric-value">{progress['weakest_level'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Page: Study Recommendations
# ─────────────────────────────────────────────────────────────
def page_recommendations():
    st.markdown('<h1 class="main-header">💡 Study Recommendations</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personalized suggestions based on your quiz performance</p>', unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login from the sidebar first.")
        return
    
    username = st.session_state.username
    chunks = st.session_state.chunks if st.session_state.chunks else None
    
    recommendations = tracker.get_study_recommendations(username, chunks)
    
    if not recommendations:
        st.info("📝 Take some quizzes first to receive personalized recommendations!")
        return
    
    # Recommended difficulty
    recommended = tracker.get_recommended_difficulty(username)
    diff_colors = {'easy': '#4ECDC4', 'medium': '#FFD93D', 'hard': '#FF6B6B'}
    rec_color = diff_colors.get(recommended, '#6C63FF')
    
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;border-color:{rec_color}40;">
        <div style="font-size:1.1rem;color:#8892B0;">Recommended Difficulty Level</div>
        <div style="font-size:2.5rem;font-weight:700;color:{rec_color};">{recommended.upper()}</div>
        <div style="color:#8892B0;margin-top:0.5rem;">Based on your performance history</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Personalized Recommendations")
    
    priority_class = {'High': 'rec-high', 'Medium': 'rec-medium', 'Low': 'rec-low'}
    priority_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    
    for rec in recommendations:
        priority = rec.get('priority', 'Medium')
        css_class = priority_class.get(priority, 'rec-medium')
        icon = priority_icon.get(priority, '🟡')
        
        st.markdown(f"""
        <div class="rec-card {css_class}">
            <div style="font-weight:600;font-size:1.05rem;">{icon} {rec['recommendation']}</div>
            <div style="color:#8892B0;margin-top:0.5rem;font-size:0.9rem;">{rec['reason']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show related chunks if available
        if rec.get('related_chunks'):
            with st.expander("📖 Related Study Material"):
                for chunk in rec['related_chunks']:
                    st.markdown(f'<div class="info-box">{chunk}</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"📝 Take {recommended.upper()} Quiz", use_container_width=True, type="primary"):
            st.session_state.page = '📝 Take Quiz'
            st.session_state.quiz_questions = []
            st.rerun()
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = '📊 Dashboard'
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Main Router
# ─────────────────────────────────────────────────────────────
page_map = {
    '📄 Upload PDF': page_upload,
    '📝 Take Quiz': page_quiz,
    '📊 Dashboard': page_dashboard,
    '💡 Recommendations': page_recommendations,
}

current_page = st.session_state.page
if current_page in page_map:
    page_map[current_page]()
else:
    page_upload()
