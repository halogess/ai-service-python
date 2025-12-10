"""
Standalone script to generate sliding window debug HTML.
Usage: python generate_debug_html.py <json_file> <output_html>
"""

import json
import sys
from collections import defaultdict

def generate_html(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = data.get('words', [])
    debug_data = data.get('debug_predictions', [])
    final_predictions = data.get('final_predictions', {})
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sliding Window Debug</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }
        th { background-color: #4CAF50; color: white; position: sticky; top: 0; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .final { background-color: #ffeb3b !important; font-weight: bold; }
        .window-col { background-color: #e3f2fd; }
        .word { max-width: 150px; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>Sliding Window Predictions Debug</h1>
    <p>Total Words: """ + str(len(words)) + """</p>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th class="word">Word</th>
"""
    
    max_window = max([d['window_idx'] for d in debug_data]) if debug_data else 0
    
    for i in range(max_window + 1):
        html += f'                <th class="window-col">W{i}</th>\n'
    
    html += """                <th class="final">Final</th>
            </tr>
        </thead>
        <tbody>
"""
    
    word_predictions = defaultdict(list)
    for d in debug_data:
        word_predictions[d['word_idx']].append(d)
    
    for word_idx in sorted(word_predictions.keys()):
        if word_idx >= len(words):
            continue
            
        word = words[word_idx]
        preds = word_predictions[word_idx]
        
        html += f'            <tr>\n'
        html += f'                <td>{word_idx}</td>\n'
        html += f'                <td class="word">{word}</td>\n'
        
        window_preds = {p['window_idx']: p for p in preds}
        for i in range(max_window + 1):
            if i in window_preds:
                p = window_preds[i]
                html += f'                <td class="window-col">{p["label"]}<br><small>{p["confidence"]:.2f}</small></td>\n'
            else:
                html += f'                <td class="window-col">-</td>\n'
        
        if str(word_idx) in final_predictions:
            fp = final_predictions[str(word_idx)]
            html += f'                <td class="final">{fp["label"]}<br><small>{fp["confidence"]:.2f}</small></td>\n'
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
    
    print(f"HTML generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_debug_html.py <json_file> <output_html>")
        sys.exit(1)
    
    generate_html(sys.argv[1], sys.argv[2])
