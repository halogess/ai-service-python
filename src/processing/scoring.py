from typing import List
from processing.rules_visual import ErrorItem

def calculate_score(errors: List[ErrorItem], total_pages: int) -> float:
    base_score = 100.0
    
    for error in errors:
        if error.severity == 'minor':
            base_score -= 1
        elif error.severity == 'major':
            base_score -= 3
        elif error.severity == 'critical':
            base_score -= 5
    
    # Ensure score doesn't go below 0
    return max(0.0, base_score)