"""
MCQ Generator Module
Generates Multiple Choice Questions from text chunks using NLP techniques.
Uses spaCy for NER, NLTK for text processing, and WordNet for distractor generation.
"""

import random
import re
import string
from typing import List, Dict, Optional, Tuple

import nltk
import spacy

# Download required NLTK data
for resource in ['punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet', 'stopwords', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    try:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        nlp = None

STOP_WORDS = set(stopwords.words('english'))


def get_difficulty_config(difficulty: str) -> Dict:
    """Get configuration parameters for a given difficulty level.
    
    Args:
        difficulty: One of 'easy', 'medium', 'hard'.
    
    Returns:
        Configuration dictionary for MCQ generation.
    """
    configs = {
        'easy': {
            'min_sentence_words': 5,
            'max_sentence_words': 20,
            'distractor_similarity': 'low',
            'questions_per_chunk': 2,
            'prefer_named_entities': True
        },
        'medium': {
            'min_sentence_words': 8,
            'max_sentence_words': 30,
            'distractor_similarity': 'medium',
            'questions_per_chunk': 2,
            'prefer_named_entities': True
        },
        'hard': {
            'min_sentence_words': 10,
            'max_sentence_words': 50,
            'distractor_similarity': 'high',
            'questions_per_chunk': 3,
            'prefer_named_entities': False
        }
    }
    return configs.get(difficulty.lower(), configs['medium'])


def extract_key_sentences(text: str, num_sentences: int = 5, difficulty: str = 'medium') -> List[str]:
    """Extract the most information-rich sentences from text.
    
    Scores sentences based on length, named entities, and noun density.
    
    Args:
        text: Source text to extract sentences from.
        num_sentences: Number of sentences to return.
        difficulty: Difficulty level affects sentence selection criteria.
    
    Returns:
        List of top-scoring sentences.
    """
    config = get_difficulty_config(difficulty)
    sentences = sent_tokenize(text)
    
    if not sentences:
        return []
    
    scored_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        word_count = len(words)
        
        # Skip sentences outside the word count range for this difficulty
        if word_count < config['min_sentence_words'] or word_count > config['max_sentence_words']:
            continue
        
        # Score components
        score = 0.0
        
        # Prefer medium-length sentences (not too short, not too long)
        ideal_length = (config['min_sentence_words'] + config['max_sentence_words']) / 2
        length_score = max(0, 1 - abs(word_count - ideal_length) / ideal_length)
        score += length_score * 2
        
        # Named entity count (using spaCy if available)
        if nlp:
            doc = nlp(sent)
            ne_count = len(doc.ents)
            score += ne_count * 3
        
        # Noun density score
        tagged = nltk.pos_tag(words)
        noun_count = sum(1 for _, tag in tagged if tag.startswith('NN'))
        score += noun_count * 1.5
        
        # Penalize questions/incomplete sentences
        if sent.endswith('?') or sent.endswith(':'):
            score *= 0.3
        
        scored_sentences.append((sent, score))
    
    # Sort by score descending and return top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored_sentences[:num_sentences]]


def extract_answer_candidates(sentence: str) -> List[Dict]:
    """Extract potential answer candidates from a sentence.
    
    Uses spaCy NER and POS tagging to find meaningful terms.
    
    Args:
        sentence: The source sentence.
    
    Returns:
        List of candidate dicts with 'answer', 'type', 'start', 'end' keys.
    """
    candidates = []
    
    # Method 1: spaCy Named Entity Recognition
    if nlp:
        doc = nlp(sentence)
        for ent in doc.ents:
            if len(ent.text.strip()) >= 2 and ent.text.lower() not in STOP_WORDS:
                candidates.append({
                    'answer': ent.text.strip(),
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
    
    # Method 2: Noun phrase extraction as fallback/supplement
    if nlp:
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # Skip if it's just a pronoun, determiner, or stopword
            if (len(chunk_text) >= 3 
                and chunk_text.lower() not in STOP_WORDS
                and not all(w.lower() in STOP_WORDS for w in chunk_text.split())):
                # Avoid duplicates with NER results
                if not any(c['answer'].lower() == chunk_text.lower() for c in candidates):
                    candidates.append({
                        'answer': chunk_text,
                        'type': 'NOUN_PHRASE',
                        'start': chunk.start_char,
                        'end': chunk.end_char
                    })
    
    # Method 3: POS-tag based extraction (fallback if spaCy found nothing)
    if not candidates:
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for i, (word, tag) in enumerate(tagged):
            if (tag.startswith('NN') and len(word) >= 3 
                and word.lower() not in STOP_WORDS
                and word not in string.punctuation):
                candidates.append({
                    'answer': word,
                    'type': 'NOUN_PHRASE',
                    'start': sentence.find(word),
                    'end': sentence.find(word) + len(word)
                })
    
    return candidates


def generate_distractors(answer: str, answer_type: str, all_candidates: List[str],
                         difficulty: str, num_distractors: int = 3) -> List[str]:
    """Generate plausible wrong answers (distractors) for an MCQ.
    
    Strategy varies by difficulty level:
    - Easy: Clearly different distractors
    - Medium: Same-type distractors from the document
    - Hard: Semantically similar distractors via WordNet
    
    Args:
        answer: The correct answer string.
        answer_type: The entity type of the answer.
        all_candidates: All available candidates from the document.
        difficulty: Difficulty level.
        num_distractors: Number of distractors to generate.
    
    Returns:
        List of distractor strings.
    """
    distractors = []
    answer_lower = answer.lower().strip()
    
    # Filter candidates to exclude the correct answer
    available = [c for c in all_candidates if c.lower().strip() != answer_lower]
    
    if difficulty.lower() == 'hard':
        # Try WordNet for semantically similar terms
        distractors.extend(_get_wordnet_distractors(answer, num_distractors))
    
    if difficulty.lower() in ('medium', 'hard'):
        # Use same-type candidates from the document
        same_type = [c for c in available if len(c) > 1]
        random.shuffle(same_type)
        for candidate in same_type:
            if candidate.lower() not in [d.lower() for d in distractors] and candidate.lower() != answer_lower:
                distractors.append(candidate)
            if len(distractors) >= num_distractors:
                break
    
    if difficulty.lower() == 'easy':
        # Pick clearly different candidates
        diff_candidates = [c for c in available 
                          if abs(len(c) - len(answer)) > 2 or c[0].lower() != answer[0].lower()]
        if not diff_candidates:
            diff_candidates = available
        random.shuffle(diff_candidates)
        for candidate in diff_candidates:
            if candidate.lower() not in [d.lower() for d in distractors] and candidate.lower() != answer_lower:
                distractors.append(candidate)
            if len(distractors) >= num_distractors:
                break
    
    # If still not enough distractors, use any available candidates
    if len(distractors) < num_distractors:
        random.shuffle(available)
        for candidate in available:
            if candidate.lower() not in [d.lower() for d in distractors] and candidate.lower() != answer_lower:
                distractors.append(candidate)
            if len(distractors) >= num_distractors:
                break
    
    # Last resort: generate generic distractors
    if len(distractors) < num_distractors:
        distractors.extend(_generate_fallback_distractors(
            answer, answer_type, num_distractors - len(distractors), distractors
        ))
    
    return distractors[:num_distractors]


def _get_wordnet_distractors(answer: str, max_count: int = 3) -> List[str]:
    """Find semantically related words using WordNet."""
    distractors = []
    answer_lower = answer.lower().replace(' ', '_')
    
    synsets = wordnet.synsets(answer_lower)
    if not synsets:
        # Try individual words if multi-word
        for word in answer.split():
            synsets.extend(wordnet.synsets(word.lower()))
    
    related_words = set()
    for syn in synsets[:3]:
        # Get hypernyms (more general terms)
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != answer.lower():
                    related_words.add(name)
        # Get hyponyms (more specific terms)
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != answer.lower():
                    related_words.add(name)
        # Get sister terms (same hypernym)
        for hyper in syn.hypernyms():
            for sister in hyper.hyponyms():
                for lemma in sister.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != answer.lower():
                        related_words.add(name)
    
    related_list = list(related_words)
    random.shuffle(related_list)
    return related_list[:max_count]


def _generate_fallback_distractors(answer: str, answer_type: str, count: int,
                                    existing: List[str]) -> List[str]:
    """Generate generic fallback distractors when other methods fail."""
    distractors = []
    existing_lower = [d.lower() for d in existing] + [answer.lower()]
    
    fallbacks = {
        'PERSON': ['Alexander Fleming', 'Marie Curie', 'Isaac Newton', 'Charles Darwin',
                   'Nikola Tesla', 'Ada Lovelace', 'Albert Einstein', 'Rosalind Franklin'],
        'DATE': ['1945', '1867', '1923', '2001', '1776', '1969', '1901', '1989'],
        'CARDINAL': ['42', '100', '256', '1024', '7', '365', '12', '99'],
        'ORG': ['UNESCO', 'World Health Organization', 'European Union', 'United Nations',
                'National Science Foundation', 'IEEE', 'MIT', 'Stanford University'],
        'GPE': ['London', 'Tokyo', 'Berlin', 'Sydney', 'Toronto', 'Paris', 'Mumbai', 'Cairo'],
        'MONEY': ['$1 million', '$500,000', '$10 billion', '$25,000'],
        'PERCENT': ['25%', '50%', '75%', '33%', '10%', '90%'],
    }
    
    # Default fallback for unknown types
    type_options = fallbacks.get(answer_type, 
                                ['Option A', 'Option B', 'Option C', 'Option D',
                                 'None of the above', 'All of the above'])
    
    random.shuffle(type_options)
    for option in type_options:
        if option.lower() not in existing_lower:
            distractors.append(option)
        if len(distractors) >= count:
            break
    
    # Absolute last resort
    while len(distractors) < count:
        generic = f"Alternative {len(distractors) + 1}"
        if generic.lower() not in existing_lower:
            distractors.append(generic)
    
    return distractors[:count]


def form_question(sentence: str, answer: str) -> str:
    """Form a question by replacing the answer in the sentence.
    
    Args:
        sentence: The source sentence.
        answer: The answer to blank out.
    
    Returns:
        A question string with the answer replaced by a blank.
    """
    # Escape special regex characters in the answer
    escaped_answer = re.escape(answer)
    
    # Check if sentence starts with the answer
    if sentence.strip().lower().startswith(answer.lower()):
        # Create a "What/Who" style question
        remainder = re.sub(escaped_answer, '', sentence, count=1, flags=re.IGNORECASE).strip()
        remainder = remainder.lstrip(',').strip()
        if remainder.endswith('.'):
            remainder = remainder[:-1]
        return f"What {remainder}?"
    
    # Standard: replace answer with blank
    question = re.sub(escaped_answer, '________', sentence, count=1, flags=re.IGNORECASE)
    
    # Clean up the question
    if not question.endswith('?'):
        if question.endswith('.'):
            question = question[:-1]
        question = "Complete the following: " + question
    
    return question


def generate_mcqs(chunks: List[Dict], difficulty: str = 'medium',
                  num_questions: int = 5) -> List[Dict]:
    """Generate Multiple Choice Questions from retrieved text chunks.
    
    This is the main entry point for MCQ generation.
    
    Args:
        chunks: List of dicts with 'chunk' and 'score' keys (from retriever).
        difficulty: One of 'easy', 'medium', 'hard'.
        num_questions: Number of questions to generate.
    
    Returns:
        List of MCQ dicts with question, options, correct_index, etc.
    """
    if not chunks:
        return []
    
    config = get_difficulty_config(difficulty)
    all_candidates_pool = []  # Global pool for distractor generation
    question_candidates = []  # (sentence, answer_dict, source_chunk)
    
    # Phase 1: Extract answer candidates from all chunks
    for chunk_data in chunks:
        chunk_text = chunk_data['chunk'] if isinstance(chunk_data, dict) else chunk_data
        key_sentences = extract_key_sentences(chunk_text, 
                                               num_sentences=config['questions_per_chunk'] + 2,
                                               difficulty=difficulty)
        
        for sentence in key_sentences:
            candidates = extract_answer_candidates(sentence)
            for candidate in candidates:
                all_candidates_pool.append(candidate['answer'])
                question_candidates.append((sentence, candidate, chunk_text))
    
    if not question_candidates:
        return []
    
    # Phase 2: Generate MCQs
    mcqs = []
    used_answers = set()  # Track used answers to avoid duplicates
    
    # Shuffle to add variety
    random.shuffle(question_candidates)
    
    for sentence, candidate, source_chunk in question_candidates:
        if len(mcqs) >= num_questions:
            break
        
        answer = candidate['answer']
        
        # Skip if we already have a question with this answer
        if answer.lower() in used_answers:
            continue
        
        # Generate distractors
        distractors = generate_distractors(
            answer=answer,
            answer_type=candidate['type'],
            all_candidates=all_candidates_pool,
            difficulty=difficulty,
            num_distractors=3
        )
        
        # Need at least 3 distractors for a 4-option MCQ
        if len(distractors) < 3:
            continue
        
        # Form the question
        question_text = form_question(sentence, answer)
        
        # Create options list and shuffle
        options = [answer] + distractors[:3]
        random.shuffle(options)
        correct_index = options.index(answer)
        
        mcq = {
            'question': question_text,
            'options': options,
            'correct_index': correct_index,
            'correct_answer': answer,
            'explanation': f"Source: \"{sentence}\"",
            'difficulty': difficulty,
            'source_chunk': source_chunk
        }
        
        mcqs.append(mcq)
        used_answers.add(answer.lower())
    
    return mcqs[:num_questions]
