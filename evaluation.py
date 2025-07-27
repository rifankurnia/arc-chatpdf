#!/usr/bin/env python3
"""
Basic evaluation system with golden Q&A pairs for testing system performance
"""
import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GoldenQA:
    """Represents a golden Q&A pair for evaluation"""
    question: str
    expected_answer: str
    category: str
    difficulty: str  # easy, medium, hard
    keywords: List[str]  # Keywords that should appear in the answer
    description: str = ""

@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single Q&A pair"""
    question: str
    expected_answer: str
    actual_answer: str
    score: float  # 0.0 to 1.0
    keywords_found: List[str]
    keywords_missing: List[str]
    category: str
    difficulty: str
    evaluation_notes: str = ""

class EvaluationSystem:
    """Basic evaluation system for golden Q&A pairs"""
    
    def __init__(self):
        self.golden_qa_pairs: List[GoldenQA] = []
        self.results: List[EvaluationResult] = []
        
    def add_golden_qa(self, qa: GoldenQA):
        """Add a golden Q&A pair to the evaluation set"""
        self.golden_qa_pairs.append(qa)
        
    def load_golden_qa_from_file(self, filepath: str):
        """Load golden Q&A pairs from a JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                qa = GoldenQA(
                    question=item['question'],
                    expected_answer=item['expected_answer'],
                    category=item['category'],
                    difficulty=item['difficulty'],
                    keywords=item.get('keywords', []),
                    description=item.get('description', '')
                )
                self.add_golden_qa(qa)
                
            logger.info(f"Loaded {len(data)} golden Q&A pairs from {filepath}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error loading golden Q&A pairs: {e}")
    
    def evaluate_answer(self, expected_answer: str, actual_answer: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate an actual answer against expected answer and keywords
        Returns: score, keywords_found, keywords_missing
        """
        # Simple keyword-based evaluation
        expected_lower = expected_answer.lower()
        actual_lower = actual_answer.lower()
        
        # Check for keyword presence
        keywords_found = []
        keywords_missing = []
        
        for keyword in keywords:
            if keyword.lower() in actual_lower:
                keywords_found.append(keyword)
            else:
                keywords_missing.append(keyword)
        
        # Calculate score based on keyword coverage
        keyword_score = len(keywords_found) / len(keywords) if keywords else 0.5
        
        # Simple text similarity (basic implementation)
        # In a real system, you might use more sophisticated metrics like BLEU, ROUGE, or semantic similarity
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())
        
        if expected_words:
            word_overlap = len(expected_words.intersection(actual_words)) / len(expected_words)
        else:
            word_overlap = 0.0
        
        # Combined score (keyword coverage + word overlap)
        final_score = (keyword_score * 0.7) + (word_overlap * 0.3)
        
        return {
            'score': min(1.0, max(0.0, final_score)),
            'keywords_found': keywords_found,
            'keywords_missing': keywords_missing,
            'word_overlap': word_overlap,
            'keyword_score': keyword_score
        }
    
    async def run_evaluation(self, query_function) -> List[EvaluationResult]:
        """
        Run evaluation on all golden Q&A pairs
        query_function: async function that takes a question and returns an answer
        """
        logger.info(f"Starting evaluation with {len(self.golden_qa_pairs)} golden Q&A pairs")
        
        results = []
        
        for i, qa in enumerate(self.golden_qa_pairs, 1):
            logger.info(f"Evaluating {i}/{len(self.golden_qa_pairs)}: {qa.question[:50]}...")
            
            try:
                # Get actual answer from the system
                actual_answer = await query_function(qa.question)
                
                # Evaluate the answer
                evaluation = self.evaluate_answer(qa.expected_answer, actual_answer, qa.keywords)
                
                # Create result
                result = EvaluationResult(
                    question=qa.question,
                    expected_answer=qa.expected_answer,
                    actual_answer=actual_answer,
                    score=evaluation['score'],
                    keywords_found=evaluation['keywords_found'],
                    keywords_missing=evaluation['keywords_missing'],
                    category=qa.category,
                    difficulty=qa.difficulty,
                    evaluation_notes=f"Keyword score: {evaluation['keyword_score']:.2f}, Word overlap: {evaluation['word_overlap']:.2f}"
                )
                
                results.append(result)
                
                logger.info(f"Score: {result.score:.2f} - Keywords found: {len(result.keywords_found)}/{len(qa.keywords)}")
                
            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                # Create failed result
                result = EvaluationResult(
                    question=qa.question,
                    expected_answer=qa.expected_answer,
                    actual_answer=f"ERROR: {str(e)}",
                    score=0.0,
                    keywords_found=[],
                    keywords_missing=qa.keywords,
                    category=qa.category,
                    difficulty=qa.difficulty,
                    evaluation_notes=f"Evaluation failed: {str(e)}"
                )
                results.append(result)
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        if not self.results:
            return {"error": "No evaluation results available"}
        
        # Calculate overall statistics
        total_questions = len(self.results)
        successful_questions = len([r for r in self.results if r.score > 0])
        failed_questions = total_questions - successful_questions
        
        # Average scores
        avg_score = sum(r.score for r in self.results) / total_questions
        avg_successful_score = sum(r.score for r in self.results if r.score > 0) / successful_questions if successful_questions > 0 else 0
        
        # Scores by category
        category_scores = {}
        for result in self.results:
            if result.category not in category_scores:
                category_scores[result.category] = []
            category_scores[result.category].append(result.score)
        
        category_averages = {
            category: sum(scores) / len(scores) 
            for category, scores in category_scores.items()
        }
        
        # Scores by difficulty
        difficulty_scores = {}
        for result in self.results:
            if result.difficulty not in difficulty_scores:
                difficulty_scores[result.difficulty] = []
            difficulty_scores[result.difficulty].append(result.score)
        
        difficulty_averages = {
            difficulty: sum(scores) / len(scores) 
            for difficulty, scores in difficulty_scores.items()
        }
        
        # Keyword analysis
        total_keywords = sum(len(r.keywords_found) + len(r.keywords_missing) for r in self.results)
        found_keywords = sum(len(r.keywords_found) for r in self.results)
        keyword_accuracy = found_keywords / total_keywords if total_keywords > 0 else 0
        
        # Top performing and worst performing questions
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)
        top_5 = sorted_results[:5]
        bottom_5 = sorted_results[-5:] if len(sorted_results) >= 5 else sorted_results
        
        report = {
            "summary": {
                "total_questions": total_questions,
                "successful_questions": successful_questions,
                "failed_questions": failed_questions,
                "success_rate": successful_questions / total_questions,
                "average_score": avg_score,
                "average_successful_score": avg_successful_score,
                "keyword_accuracy": keyword_accuracy
            },
            "category_performance": category_averages,
            "difficulty_performance": difficulty_averages,
            "top_performing": [
                {
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                    "score": r.score,
                    "category": r.category,
                    "difficulty": r.difficulty
                } for r in top_5
            ],
            "worst_performing": [
                {
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                    "score": r.score,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "notes": r.evaluation_notes
                } for r in bottom_5
            ],
            "detailed_results": [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "score": r.score,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "keywords_found": r.keywords_found,
                    "keywords_missing": r.keywords_missing,
                    "notes": r.evaluation_notes
                } for r in self.results
            ]
        }
        
        return report
    
    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file"""
        try:
            report = self.generate_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self):
        """Print a summary of evaluation results"""
        if not self.results:
            print("No evaluation results available")
            return
        
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Successful Questions: {summary['successful_questions']}")
        print(f"Failed Questions: {summary['failed_questions']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Average Score: {summary['average_score']:.3f}")
        print(f"Average Successful Score: {summary['average_successful_score']:.3f}")
        print(f"Keyword Accuracy: {summary['keyword_accuracy']:.2%}")
        
        print(f"\nCategory Performance:")
        for category, score in report["category_performance"].items():
            print(f"  {category}: {score:.3f}")
        
        print(f"\nDifficulty Performance:")
        for difficulty, score in report["difficulty_performance"].items():
            print(f"  {difficulty}: {score:.3f}")
        
        print(f"\nTop 3 Performing Questions:")
        for i, item in enumerate(report["top_performing"][:3], 1):
            print(f"  {i}. Score: {item['score']:.3f} - {item['question']}")
        
        print(f"\nBottom 3 Performing Questions:")
        for i, item in enumerate(report["worst_performing"][:3], 1):
            print(f"  {i}. Score: {item['score']:.3f} - {item['question']}")

def create_sample_golden_qa():
    """Create sample golden Q&A pairs for testing"""
    sample_qa = [
        {
            "question": "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?",
            "expected_answer": "SimpleDDL-MD-Chat achieved the highest zero-shot accuracy of 71.6% on Spider",
            "category": "performance_metrics",
            "difficulty": "medium",
            "keywords": ["SimpleDDL-MD-Chat", "71.6%", "Spider", "zero-shot"],
            "description": "Tests ability to find specific performance metrics from research papers"
        },
        {
            "question": "What other prompt templates were tested in Zhang et al. (2024)?",
            "expected_answer": "The paper tested DDL-HTML-Chat, DDL-HTML-Complete, DDL-MD-Chat, DDL-MD-Complete, DDL-Coding-Chat, DDL-Coding-Complete, SimpleDDL-MD-Chat, and SimpleDDL-MD-Complete",
            "category": "methodology",
            "difficulty": "easy",
            "keywords": ["DDL-HTML-Chat", "DDL-MD-Chat", "DDL-Coding-Chat", "SimpleDDL-MD-Complete"],
            "description": "Tests ability to list methodology details from research papers"
        },
        {
            "question": "What is the latest news about OpenAI?",
            "expected_answer": "OpenAI has made several recent announcements including new models and partnerships",
            "category": "current_events",
            "difficulty": "easy",
            "keywords": ["OpenAI", "recent", "announcements", "models"],
            "description": "Tests web search capability for current events"
        },
        {
            "question": "How does the performance of DDL-MD-Chat compare to SimpleDDL-MD-Chat?",
            "expected_answer": "SimpleDDL-MD-Chat performed better than DDL-MD-Chat with higher accuracy scores",
            "category": "comparison",
            "difficulty": "hard",
            "keywords": ["SimpleDDL-MD-Chat", "DDL-MD-Chat", "better", "higher", "accuracy"],
            "description": "Tests ability to compare different methods from research papers"
        }
    ]
    
    return sample_qa

async def main():
    """Main function to run evaluation"""
    print("🚀 Starting Basic Evaluation System")
    print("="*60)
    
    # Create evaluation system
    evaluator = EvaluationSystem()
    
    # Add sample golden Q&A pairs
    sample_qa = create_sample_golden_qa()
    for qa_data in sample_qa:
        qa = GoldenQA(**qa_data)
        evaluator.add_golden_qa(qa)
    
    # Import the query function
    try:
        from main import run_query
    except ImportError as e:
        print(f"❌ Error importing run_query: {e}")
        return 1
    
    # Run evaluation
    print(f"📝 Running evaluation on {len(evaluator.golden_qa_pairs)} questions...")
    results = await evaluator.run_evaluation(run_query)
    
    # Generate and display report
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    
    print("\n✅ Evaluation completed!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 