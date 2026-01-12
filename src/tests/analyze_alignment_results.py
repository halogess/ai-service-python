"""
Analyze actual alignment results from JSON files
"""

import json
import os
import glob

def analyze_alignment_files():
    """Find and analyze alignment result JSON files"""
    
    # Search for alignment_results.json files
    pattern = "**/alignment_results.json"
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("No alignment_results.json files found!")
        print("\nSearching in common locations...")
        
        common_paths = [
            "outputs/**/alignment_results.json",
            "data/**/alignment_results.json",
            "results/**/alignment_results.json"
        ]
        
        for path in common_paths:
            files.extend(glob.glob(path, recursive=True))
    
    if not files:
        print("Still no files found. Please provide path manually.")
        return
    
    print(f"Found {len(files)} alignment result file(s):\n")
    
    for file_path in files[:5]:  # Analyze first 5 files
        print("="*80)
        print(f"FILE: {file_path}")
        print("="*80)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                print(f"Total pages: {len(data)}")
                
                for page_result in data[:2]:  # Analyze first 2 pages
                    page_num = page_result.get('page', '?')
                    print(f"\n--- PAGE {page_num} ---")
                    print(f"Success: {page_result.get('success')}")
                    
                    alignments = page_result.get('alignments', [])
                    unaligned = page_result.get('unaligned_pdf_units', [])
                    header_footer = page_result.get('header_footer_units', [])
                    
                    print(f"Alignments: {len(alignments)}")
                    print(f"Unaligned: {len(unaligned)}")
                    print(f"Header/Footer: {len(header_footer)}")
                    
                    if alignments:
                        print(f"\nFirst alignment:")
                        first = alignments[0]
                        print(f"  element_id: {first.get('element_id')}")
                        print(f"  merged_bbox: {first.get('merged_bbox')}")
                        print(f"  matched_pdf_units: {len(first.get('matched_pdf_units', []))}")
                        
                        if first.get('matched_pdf_units'):
                            unit = first['matched_pdf_units'][0]
                            print(f"  First unit bbox: {unit.get('bbox')}")
                    else:
                        print("  ⚠️ NO ALIGNMENTS!")
                    
                    if unaligned:
                        print(f"\nFirst unaligned:")
                        first_un = unaligned[0]
                        if isinstance(first_un, dict):
                            print(f"  Type: dict")
                            print(f"  bbox: {first_un.get('bbox')}")
                        else:
                            print(f"  Type: {type(first_un)}")
                            print(f"  ⚠️ NOT A DICT! Value: {first_un}")
                    
                    if header_footer:
                        print(f"\nFirst header/footer:")
                        first_hf = header_footer[0]
                        if isinstance(first_hf, dict):
                            print(f"  Type: dict")
                            print(f"  bbox: {first_hf.get('bbox')}")
                        else:
                            print(f"  Type: {type(first_hf)}")
                            print(f"  ⚠️ NOT A DICT! Value: {first_hf}")
            
            print()
            
        except Exception as e:
            print(f"Error reading file: {e}")
            print()

if __name__ == "__main__":
    analyze_alignment_files()
