"""Formula Y-gap filtering to remove tokens from different formulas"""


def filter_formula_tokens_by_y_gap(element_groups):
    """Filter out tokens that are too far vertically from main formula line"""
    
    for elem_id in element_groups:
        for cell_idx in element_groups[elem_id]:
            words = element_groups[elem_id][cell_idx]
            if len(words) <= 1:
                continue
            
            # Cek apakah ini formula
            is_formula = any(w.get('is_formula', False) for w in words)
            if not is_formula:
                continue
            
            # Jangan filter jika elemen lintas halaman
            pages = set(w['page'] for w in words)
            if len(pages) > 1:
                continue
            
            # Hitung Y midpoint untuk setiap token dan sort
            word_y_pairs = [(w, (w['bbox']['y0'] + w['bbox']['y1']) / 2) for w in words]
            word_y_pairs.sort(key=lambda x: x[1])  # Sort by Y
            
            # Hitung tinggi font rata-rata
            heights = [w['bbox']['y1'] - w['bbox']['y0'] for w in words]
            avg_height = sum(heights) / len(heights) if heights else 15
            
            # Cari gap yang signifikan (> 2x avg_height = kemungkinan formula berbeda)
            # Gap 2x karena formula dengan pecahan bisa punya gap 1.5x untuk fraction line
            gap_threshold = avg_height * 2.0
            
            cutoff_y = None
            for i in range(len(word_y_pairs) - 1):
                current_y = word_y_pairs[i][1]
                next_y = word_y_pairs[i + 1][1]
                gap = next_y - current_y
                
                if gap > gap_threshold:
                    # Gap besar ditemukan - token setelah ini milik elemen lain
                    cutoff_y = current_y + gap_threshold / 2  # Cutoff di tengah gap
                    break
            
            if cutoff_y is not None:
                # Filter token yang berada sebelum cutoff
                filtered_words = [w for w, y in word_y_pairs if y <= cutoff_y]
                
                if len(filtered_words) < len(words):
                    element_groups[elem_id][cell_idx] = filtered_words
