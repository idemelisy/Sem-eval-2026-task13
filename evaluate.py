"""
Evaluation script for SemEval-2026 Task 13 Part A
Evaluates trained models on test/validation data
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import argparse
import json
import os
from tqdm import tqdm

from data_loader import CodeDataset, load_data


def evaluate_model(model, tokenizer, dataloader, device):
    """Evaluate model and return metrics"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(true_labels, predictions)
    
    # Classification report
    report = classification_report(
        true_labels, predictions,
        target_names=['Human-written', 'Machine-generated'],
        output_dict=True
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model for SemEval-2026 Task 13 Part A')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (CSV or JSON)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results (JSON)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading test data...')
    test_texts, test_labels = load_data(args.test_data)
    print(f'Test samples: {len(test_texts)}')
    
    # Load tokenizer and model
    print(f'Loading model from: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    
    # Create dataset
    test_dataset = CodeDataset(test_texts, test_labels, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print('\nEvaluating model...')
    results = evaluate_model(model, tokenizer, test_loader, device)
    
    # Print results
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'Accuracy: {results["accuracy"]:.4f}')
    print(f'\nMacro Average:')
    print(f'  Precision: {results["precision_macro"]:.4f}')
    print(f'  Recall: {results["recall_macro"]:.4f}')
    print(f'  F1-Score: {results["f1_macro"]:.4f}')
    print(f'\nWeighted Average:')
    print(f'  Precision: {results["precision_weighted"]:.4f}')
    print(f'  Recall: {results["recall_weighted"]:.4f}')
    print(f'  F1-Score: {results["f1_weighted"]:.4f}')
    print(f'\nPer Class Metrics:')
    print(f'  Human-written:')
    print(f'    Precision: {results["precision_per_class"][0]:.4f}')
    print(f'    Recall: {results["recall_per_class"][0]:.4f}')
    print(f'    F1-Score: {results["f1_per_class"][0]:.4f}')
    print(f'    Support: {results["support_per_class"][0]}')
    print(f'  Machine-generated:')
    print(f'    Precision: {results["precision_per_class"][1]:.4f}')
    print(f'    Recall: {results["recall_per_class"][1]:.4f}')
    print(f'    F1-Score: {results["f1_per_class"][1]:.4f}')
    print(f'    Support: {results["support_per_class"][1]}')
    print(f'\nConfusion Matrix:')
    print(f'  [[{results["confusion_matrix"][0][0]}, {results["confusion_matrix"][0][1]}],')
    print(f'   [{results["confusion_matrix"][1][0]}, {results["confusion_matrix"][1][1]}]]')
    print('='*50)
    
    # Save results
    if args.output_file:
        # Remove predictions and probabilities for smaller file size (optional)
        results_to_save = results.copy()
        # Uncomment to exclude predictions/probabilities:
        # results_to_save.pop('predictions', None)
        # results_to_save.pop('probabilities', None)
        
        with open(args.output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f'\nResults saved to: {args.output_file}')


if __name__ == '__main__':
    main()

