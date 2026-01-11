from collections import defaultdict
from models import TestingHistory, TestingGroundTruth, TestingPrediction

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_confusion_matrix(history_id):
    history = TestingHistory.query.get_or_404(history_id)
    doc_id = history.testing_dokumen_id
    
    gt_data = TestingGroundTruth.query.filter_by(testing_dokumen_id=doc_id).all()
    if not gt_data:
        return {'error': 'No ground truth found. Please create ground truth first.'}
    
    pred_data = TestingPrediction.query.filter_by(testing_history_id=history_id).all()
    
    all_labels = set()
    for gt in gt_data:
        all_labels.add(gt.testing_ground_truth_label)
    for pred in pred_data:
        all_labels.add(pred.testing_prediction_label)
    
    possible_labels = ['Title', 'Text', 'List-item', 'Table', 'Picture', 'Caption', 
                      'Section-header', 'Page-header', 'Page-footer', 'Footnote', 'Formula']
    
    labels = sorted(list(all_labels))
    for label in possible_labels:
        if label not in labels:
            labels.append(label)
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    
    gt_by_page = defaultdict(list)
    pred_by_page = defaultdict(list)
    
    for gt in gt_data:
        gt_by_page[gt.testing_ground_truth_page].append(gt)
    for pred in pred_data:
        pred_by_page[pred.testing_prediction_page].append(pred)
    
    matched_gt = set()
    for page in pred_by_page.keys():
        for pred in pred_by_page[page]:
            best_iou = 0
            best_gt = None
            
            for gt in gt_by_page.get(page, []):
                if id(gt) in matched_gt:
                    continue
                iou = calculate_iou(pred.testing_prediction_bbox, gt.testing_ground_truth_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou > 0.5 and best_gt:
                matched_gt.add(id(best_gt))
                pred_idx = label_to_idx[pred.testing_prediction_label]
                gt_idx = label_to_idx[best_gt.testing_ground_truth_label]
                matrix[pred_idx][gt_idx] += 1
    
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    precisions = []
    recalls = []
    for i in range(n):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(n))
        
        if row_sum > 0 or col_sum > 0:
            precision = (matrix[i][i] / row_sum * 100) if row_sum > 0 else 0
            recall = (matrix[i][i] / col_sum * 100) if col_sum > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    f1score = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'labels': labels,
        'matrix': matrix,
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1score': f1score
    }
