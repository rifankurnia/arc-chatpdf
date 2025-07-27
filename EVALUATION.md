# Evaluation System Documentation

## Overview

The evaluation system provides a comprehensive way to test the performance of the Chat PDF system using golden Q&A pairs. It measures how well the system answers specific questions compared to expected answers.

## Features

### ✅ **Core Capabilities**

1. **Golden Q&A Pairs**: Pre-defined question-answer pairs for testing
2. **Keyword-based Evaluation**: Checks for specific keywords in responses
3. **Text Similarity Scoring**: Measures word overlap between expected and actual answers
4. **Category-based Analysis**: Groups questions by type (performance_metrics, methodology, etc.)
5. **Difficulty-based Analysis**: Groups questions by difficulty (easy, medium, hard)
6. **Comprehensive Reporting**: Detailed performance metrics and analysis

### ✅ **Evaluation Metrics**

- **Success Rate**: Percentage of questions answered successfully
- **Average Score**: Overall performance score (0.0 to 1.0)
- **Keyword Accuracy**: Percentage of expected keywords found in answers
- **Category Performance**: Performance breakdown by question category
- **Difficulty Performance**: Performance breakdown by difficulty level

## Usage

### **Command Line Evaluation**

```bash
# Run evaluation with sample golden Q&A pairs
python3.11 run_evaluation.py

# Run evaluation directly
python3.11 evaluation.py
```

### **REST API Evaluation**

```bash
# Run evaluation via API
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json"
```

### **Programmatic Usage**

```python
from evaluation import EvaluationSystem, GoldenQA

# Create evaluation system
evaluator = EvaluationSystem()

# Add golden Q&A pairs
qa = GoldenQA(
    question="Which prompt template gave the highest zero-shot accuracy?",
    expected_answer="SimpleDDL-MD-Chat achieved 71.6% accuracy",
    category="performance_metrics",
    difficulty="medium",
    keywords=["SimpleDDL-MD-Chat", "71.6%", "accuracy"]
)
evaluator.add_golden_qa(qa)

# Run evaluation
results = await evaluator.run_evaluation(run_query)

# Generate report
report = evaluator.generate_report()
evaluator.print_summary()
```

## Golden Q&A Format

### **JSON Structure**

```json
{
  "question": "The question to test",
  "expected_answer": "The expected answer",
  "category": "performance_metrics|methodology|current_events|comparison|paper_overview|clarification",
  "difficulty": "easy|medium|hard",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "description": "Optional description of what this tests"
}
```

### **Sample Golden Q&A Pairs**

The system includes 8 sample golden Q&A pairs covering:

1. **Performance Metrics**: Specific accuracy numbers and model performance
2. **Methodology**: Research methods and experimental details
3. **Current Events**: Web search capabilities for recent information
4. **Comparisons**: Comparing different methods or approaches
5. **Paper Overview**: Understanding paper contributions and findings
6. **Clarification**: Handling ambiguous queries

## Evaluation Algorithm

### **Scoring Method**

The evaluation uses a combined scoring approach:

1. **Keyword Coverage (70% weight)**:
   - Checks for presence of expected keywords in the answer
   - Score = (keywords found) / (total keywords)

2. **Word Overlap (30% weight)**:
   - Measures word-level similarity between expected and actual answers
   - Score = (common words) / (expected words)

3. **Final Score**: `(keyword_score * 0.7) + (word_overlap * 0.3)`

### **Example Scoring**

```
Question: "Which prompt template gave the highest zero-shot accuracy?"
Expected: "SimpleDDL-MD-Chat achieved 71.6% accuracy"
Keywords: ["SimpleDDL-MD-Chat", "71.6%", "accuracy"]

Actual Answer: "SimpleDDL-MD-Chat achieved the highest zero-shot accuracy of 71.6%"
- Keywords found: 3/3 (100%)
- Word overlap: 6/8 (75%)
- Final score: (1.0 * 0.7) + (0.75 * 0.3) = 0.925
```

## Output Files

### **evaluation_results.json**

Contains detailed evaluation results including:

- **Summary**: Overall performance metrics
- **Category Performance**: Scores by question category
- **Difficulty Performance**: Scores by difficulty level
- **Top/Bottom Performing**: Best and worst questions
- **Detailed Results**: Complete evaluation data for each question

### **Sample Report Structure**

```json
{
  "summary": {
    "total_questions": 8,
    "successful_questions": 8,
    "failed_questions": 0,
    "success_rate": 1.0,
    "average_score": 0.575,
    "keyword_accuracy": 0.636
  },
  "category_performance": {
    "performance_metrics": 0.921,
    "methodology": 0.605,
    "current_events": 0.716
  },
  "difficulty_performance": {
    "easy": 0.494,
    "medium": 0.921,
    "hard": 0.280
  }
}
```

## Adding Custom Golden Q&A Pairs

### **Method 1: JSON File**

1. Create or edit `golden_qa_pairs.json`
2. Add new Q&A pairs following the JSON format
3. Run evaluation to test

### **Method 2: Programmatic**

```python
from evaluation import EvaluationSystem, GoldenQA

evaluator = EvaluationSystem()

# Add custom Q&A pair
custom_qa = GoldenQA(
    question="Your custom question here?",
    expected_answer="Expected answer with key information",
    category="your_category",
    difficulty="medium",
    keywords=["important", "keywords", "to", "check"],
    description="What this question tests"
)

evaluator.add_golden_qa(custom_qa)
```

## Best Practices

### **Creating Effective Golden Q&A Pairs**

1. **Specific Keywords**: Include exact terms that should appear in answers
2. **Realistic Expectations**: Expected answers should be achievable
3. **Diverse Categories**: Cover different types of questions
4. **Varying Difficulty**: Include easy, medium, and hard questions
5. **Clear Context**: Questions should be unambiguous

### **Interpreting Results**

1. **High Scores (>0.8)**: System performs well on this type of question
2. **Medium Scores (0.4-0.8)**: Room for improvement
3. **Low Scores (<0.4)**: Significant issues with this question type
4. **Category Analysis**: Identify strengths and weaknesses by category
5. **Difficulty Analysis**: Understand performance across difficulty levels

## Troubleshooting

### **Common Issues**

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check that `golden_qa_pairs.json` exists
3. **API Errors**: Verify API keys are set correctly
4. **Low Scores**: Review expected answers and keywords for accuracy

### **Debugging**

```bash
# Test individual components
curl -X POST "http://localhost:8000/debug/routing" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your test question"}'

# Check API health
curl -X GET "http://localhost:8000/health"
```

## Future Enhancements

### **Planned Improvements**

1. **Semantic Similarity**: More sophisticated text similarity metrics
2. **BLEU/ROUGE Scores**: Standard NLP evaluation metrics
3. **Human Evaluation**: Integration with human judgment
4. **Automated Testing**: CI/CD integration for continuous evaluation
5. **Performance Tracking**: Historical performance monitoring 