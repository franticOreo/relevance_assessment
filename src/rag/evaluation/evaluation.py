from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from pipeline_evaluator import PipelineEvaluator
from pathlib import Path
from rag.loader.document_loader import get_default_documents

def load_test_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """Load and validate the test dataset."""
    df = pd.read_csv(csv_path)
    
    # Define required columns based on our dataset schema
    required_columns = ['user_input', 'reference', 'reference_contexts']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    
    # Convert DataFrame to list of dictionaries
    dataset = df.to_dict('records')
    
    # Parse the reference_contexts string into a list
    for sample in dataset:
        # Convert string representation of list to actual list
        if isinstance(sample['reference_contexts'], str):
            # Safely evaluate the string as a Python expression
            sample['reference_contexts'] = eval(sample['reference_contexts'])
    
    return dataset

def plot_results(results: Dict[str, Any], metric_name: str, output_dir: Path) -> None:
    """Create a bar plot for a specific metric across all pipeline variants."""
    variants = list(results.keys())
    values = [r["summary"][metric_name] for r in results.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(variants, values)
    plt.xticks(rotation=45)
    plt.title(f"{metric_name} Across Pipeline Variants")
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{metric_name}_comparison.png")
    plt.close()

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "results"
    
    # Load test dataset
    test_dataset = load_test_dataset(str(data_dir / "ragas_testset.csv"))
    
    # Load documents
    documents = get_default_documents()
    
    # Initialize evaluator with both required arguments
    evaluator = PipelineEvaluator(
        test_dataset=test_dataset,
        documents=documents
    )
    
    # Run evaluations
    results = evaluator.run_all_evaluations()
    
    # Plot results for each metric
    metrics_to_plot = [
        "avg_precision",
        "avg_recall",
        "avg_mrr",
        "avg_rouge_l",
        "avg_bleu",
        "avg_judge_score_normalized",
        "composite_score"
    ]
    
    for metric in metrics_to_plot:
        plot_results(results, metric, output_dir)
    
    # Print summary results
    print("\nEvaluation Results Summary:")
    for variant, result in results.items():
        print(f"\n{variant}:")
        for metric, value in result["summary"].items():
            print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    main()