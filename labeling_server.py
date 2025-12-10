"""
Web interface for labeling LayoutLM predictions.
Usage: python labeling_server.py <layoutlm_results.json>
"""

from flask import Flask, render_template_string, request, jsonify
import json
import sys
import os

app = Flask(__name__)

# Global data
data = {}
current_file = ""

LABELS = ["Title", "Text", "Section-header", "List-item", "Table", "Picture", "Caption", "Page-header", "Page-footer", "Footnote", "Formula"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LayoutLM Labeling Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .stats { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .stats span { margin-right: 20px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; position: sticky; top: 0; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .word-cell { max-width: 200px; word-wrap: break-word; font-weight: bold; }
        .pred-cell { color: #1976d2; }
        select { padding: 8px; border: 2px solid #ddd; border-radius: 4px; font-size: 14px; width: 100%; }
        select:focus { border-color: #4CAF50; outline: none; }
        .match { background-color: #c8e6c9 !important; }
        .mismatch { background-color: #ffcdd2 !important; }
        .buttons { margin: 20px 0; }
        button { background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; margin-right: 10px; }
        button:hover { background: #45a049; }
        .export-btn { background: #2196F3; }
        .export-btn:hover { background: #0b7dda; }
        .progress { font-size: 18px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏷️ LayoutLM Labeling Tool</h1>
        
        <div class="stats">
            <span>📄 Total Words: <span id="total">0</span></span>
            <span>✅ Labeled: <span id="labeled">0</span></span>
            <span>⏳ Remaining: <span id="remaining">0</span></span>
            <span class="progress">Progress: <span id="progress">0%</span></span>
        </div>

        <div class="buttons">
            <button onclick="saveLabels()">💾 Save Labels</button>
            <button onclick="exportGroundTruth()" class="export-btn">📥 Export Ground Truth</button>
            <button onclick="autoAcceptAll()">✓ Accept All Predictions</button>
        </div>

        <table id="labelTable">
            <thead>
                <tr>
                    <th style="width: 50px;">ID</th>
                    <th style="width: 200px;">Word</th>
                    <th style="width: 150px;">Prediction</th>
                    <th style="width: 150px;">Confidence</th>
                    <th style="width: 200px;">Ground Truth Label</th>
                </tr>
            </thead>
            <tbody id="tableBody">
            </tbody>
        </table>
    </div>

    <script>
        let data = {{ data | tojson }};
        
        function renderTable() {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            let labeled = 0;
            
            data.predictions.forEach((pred, idx) => {
                const row = document.createElement('tr');
                const gtLabel = data.ground_truth ? data.ground_truth[idx] : null;
                
                if (gtLabel) labeled++;
                
                const match = gtLabel && gtLabel === pred.label;
                if (gtLabel) {
                    row.className = match ? 'match' : 'mismatch';
                }
                
                row.innerHTML = `
                    <td>${idx}</td>
                    <td class="word-cell">${data.words[idx] || '-'}</td>
                    <td class="pred-cell">${pred.label}</td>
                    <td>${pred.confidence.toFixed(3)}</td>
                    <td>
                        <select id="label_${idx}" onchange="updateLabel(${idx}, this.value)">
                            <option value="">-- Select --</option>
                            ${LABELS.map(l => `<option value="${l}" ${gtLabel === l ? 'selected' : ''}>${l}</option>`).join('')}
                        </select>
                    </td>
                `;
                tbody.appendChild(row);
            });
            
            updateStats(labeled);
        }
        
        function updateLabel(idx, label) {
            if (!data.ground_truth) data.ground_truth = [];
            data.ground_truth[idx] = label;
            renderTable();
        }
        
        function updateStats(labeled) {
            const total = data.predictions.length;
            const remaining = total - labeled;
            const progress = ((labeled / total) * 100).toFixed(1);
            
            document.getElementById('total').textContent = total;
            document.getElementById('labeled').textContent = labeled;
            document.getElementById('remaining').textContent = remaining;
            document.getElementById('progress').textContent = progress + '%';
        }
        
        function saveLabels() {
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(result => {
                alert('✅ Labels saved successfully!');
            });
        }
        
        function exportGroundTruth() {
            fetch('/export')
            .then(r => r.json())
            .then(result => {
                const blob = new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ground_truth.json';
                a.click();
                alert('✅ Ground truth exported!');
            });
        }
        
        function autoAcceptAll() {
            if (confirm('Accept all predictions as ground truth?')) {
                data.ground_truth = data.predictions.map(p => p.label);
                renderTable();
            }
        }
        
        const LABELS = {{ labels | tojson }};
        renderTable();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, data=data, labels=LABELS)

@app.route('/save', methods=['POST'])
def save():
    global data
    data = request.json
    
    # Save to file
    output_file = current_file.replace('.json', '_labeled.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return jsonify({"status": "success", "file": output_file})

@app.route('/export')
def export():
    gt = {
        "page_1": {
            "words": data.get('words', []),
            "labels": data.get('ground_truth', [])
        }
    }
    return jsonify(gt)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python labeling_server.py <layoutlm_results.json>")
        sys.exit(1)
    
    current_file = sys.argv[1]
    
    with open(current_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Convert to labeling format
    if isinstance(results, list) and len(results) > 0:
        page = results[0]  # First page
        data = {
            'words': page.get('words', []),
            'predictions': page.get('predictions', []),
            'ground_truth': []
        }
    else:
        data = results
    
    print("=" * 60)
    print("🚀 Labeling Server Started")
    print("=" * 60)
    print(f"📂 File: {current_file}")
    print(f"📊 Words: {len(data.get('words', []))}")
    print(f"🌐 Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
