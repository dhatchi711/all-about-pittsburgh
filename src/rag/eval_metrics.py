import evaluate
import string
import json

squad_metric = evaluate.load("squad")

pred_path = '/Users/kgdhatchi/Desktop/11711_RAG_A2-1/src/eval_dhakshin_output.txt'
ref_path = '/Users/kgdhatchi/Desktop/11711_RAG_A2-1/src/eval_alex_output.txt'

pred_lines = []
ref_lines = []

with open(pred_path, 'r') as file:
    pred_lines = file.read().splitlines()

with open(ref_path, 'r') as file:
    ref_lines = file.read().splitlines()

print(pred_lines)
print(ref_lines)
print(len(pred_lines))
print(len(ref_lines))

predictions = []
for i, line in enumerate(pred_lines):
    predictions.append({
        'id': str(i),
        'prediction_text': line.strip()
    })

print(predictions[:3])
print(len(predictions))

references = []
for i, line in enumerate(ref_lines):
    references.append({
        'id': str(i),
        'answers': {
            'text': [line.strip()],
            'answer_start': [0]
        }
    })
print("\nSample references:")
print(references[:3])
print(f"Total references: {len(references)}")



def normalize_text(text):
    return ''.join(ch for ch in text.lower() if ch not in set(string.punctuation)).strip()

def calculate_precision_recall_f1(predictions, references):
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_count = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_text(pred['prediction_text']).split()
        
        ref_texts = [normalize_text(ans).split() for ans in ref['answers']['text']]
        
        max_f1 = 0.0
        max_precision = 0.0
        max_recall = 0.0
        
        for ref_tokens in ref_texts:
            common_tokens = set(pred_tokens) & set(ref_tokens)
            num_common = len(common_tokens)
            
            if num_common == 0:
                continue
            
            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            if f1 > max_f1:
                max_f1 = f1
                max_precision = precision
                max_recall = recall
        
        total_precision += max_precision
        total_recall += max_recall
        total_f1 += max_f1
        total_count += 1
    
    avg_precision = (total_precision / total_count) * 100 if total_count > 0 else 0
    avg_recall = (total_recall / total_count) * 100 if total_count > 0 else 0
    avg_f1 = (total_f1 / total_count) * 100 if total_count > 0 else 0
    
    return avg_precision, avg_recall, avg_f1

results = squad_metric.compute(predictions=predictions, references=references)

avg_precision, avg_recall, avg_f1 = calculate_precision_recall_f1(predictions, references)

# Display the results
print("Results from 'evaluate' library:")
print("Exact Match (EM): {:.2f}%".format(results['exact_match']))
print("F1 Score: {:.2f}%\n".format(results['f1']))

print("Custom Calculated Metrics:")
print("Average Precision: {:.2f}%".format(avg_precision))
print("Average Recall: {:.2f}%".format(avg_recall))
print("Average F1 Score: {:.2f}%".format(avg_f1))
