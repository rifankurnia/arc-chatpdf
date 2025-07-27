#!/usr/bin/env python3
"""
Simple script to run the evaluation system
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main():
    """Run the evaluation system"""
    print("🚀 Running Evaluation System")
    print("="*50)
    
    try:
        # Import and run evaluation
        from evaluation import EvaluationSystem, GoldenQA
        
        # Create evaluation system
        evaluator = EvaluationSystem()
        
        # Load golden Q&A pairs from file
        evaluator.load_golden_qa_from_file("golden_qa_pairs.json")
        
        if not evaluator.golden_qa_pairs:
            print("❌ No golden Q&A pairs loaded. Please check the JSON file.")
            return 1
        
        print(f"📝 Loaded {len(evaluator.golden_qa_pairs)} golden Q&A pairs")
        
        # Import the query function
        from main import run_query
        
        # Run evaluation
        print("🔄 Running evaluation...")
        results = await evaluator.run_evaluation(run_query)
        
        # Display results
        evaluator.print_summary()
        
        # Save detailed results
        evaluator.save_results("evaluation_results.json")
        
        print("\n✅ Evaluation completed!")
        print("📊 Results saved to: evaluation_results.json")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 