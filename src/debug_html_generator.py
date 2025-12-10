def generate_sliding_window_debug_html(words, debug_data, final_predictions, output_path):
    """
    Generate HTML table showing predictions per window and final result.
    
    Args:
        words: List of word strings
        debug_data: List of dicts with {word_idx, window_idx, label, confidence}
        final_predictions: Dict of {word_idx: (label_id, confidence)}
        output_path: Path to save HTML file
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sliding Window Debug</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .final { background-color: #ffeb3b !important; font-weight: bold; }
        .window-col { background-color: #e3f2fd; }
    </style>
</head>
<body>
    <h1>Sliding Window Predictions Debug</h1>
    <table>
        <thead>
            <tr>
                <th>Word ID</th>
                <th>Word</th>
"""
    
    # Get max windows
    max_window = max([d['window_idx'] for d in debug_data]) if debug_data else 0
    
    # Add window columns
    for i in range(max_window + 1):
        html += f'                <th class="window-col">Window {i}</th>\n'
    
    html += """                <th class="final">Final Prediction</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Group by word_idx
    from collections import defaultdict
    word_predictions = defaultdict(list)
    for d in debug_data:
        word_predictions[d['word_idx']].append(d)
    
    # Generate rows
    for word_idx in sorted(word_predictions.keys()):
        if word_idx >= len(words):
            continue
            
        word = words[word_idx]
        preds = word_predictions[word_idx]
        
        html += f'            <tr>\n'
        html += f'                <td>{word_idx}</td>\n'
        html += f'                <td>{word}</td>\n'
        
        # Window predictions
        window_preds = {p['window_idx']: p for p in preds}
        for i in range(max_window + 1):
            if i in window_preds:
                p = window_preds[i]
                html += f'                <td class="window-col">{p["label"]}<br><small>({p["confidence"]:.3f})</small></td>\n'
            else:
                html += f'                <td class="window-col">-</td>\n'
        
        # Final prediction
        if word_idx in final_predictions:
            label_id, conf = final_predictions[word_idx]
            from layoutlm_processor import model
            label = model.config.id2label.get(label_id, "UNKNOWN")
            html += f'                <td class="final">{label}<br><small>({conf:.3f})</small></td>\n'
        else:
            html += f'                <td class="final">-</td>\n'
        
        html += f'            </tr>\n'
    
    html += """        </tbody>
    </table>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
