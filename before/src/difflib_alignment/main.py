"""Main alignment function"""

import sys
import fitz

from .docx_tokenizer import build_docx_tokens
from .pdf_tokenizer import build_pdf_tokens
from .alignment_core import perform_global_alignment, group_aligned_tokens
from .formula_gap_filter import filter_formula_tokens_by_y_gap
from .shape_fallback_aligner import align_unaligned_shapes
from .table_cell_fallback import align_unaligned_table_cells
from .image_extractor import get_pdf_images
from .image_aligner import align_table_cell_images
from .image_alignment_workflow import collect_image_items, align_standalone_images
from .image_cell_estimator import estimate_image_only_cell_positions, get_table_columns_from_docx
from .element_builder import build_element_metadata, build_shape_elements
from .table_builder import build_table_container, build_table_cells
from .table_column_aligner import align_table_columns_x
from .container_cleanup import cleanup_container_bboxes
from .non_table_builder import build_non_table_elements
from .unaligned_fallback import add_unaligned_elements_fallback
from .result_builder import build_alignment_result
from .table_structure_cache import get_table_structure_from_pdf


def align_document(pdf_path: str, elements: list, log_file=None) -> dict:
    """Align OpenXML elements dengan PDF menggunakan global difflib alignment"""
    
    if log_file:
        log_file.write("\n=== ALIGNMENT TRACE LOG ===\n")
        log_file.flush()
    sys.stderr.write("\n=== ALIGNMENT TRACE LOG ===\n")
    sys.stderr.flush()

    # Extract images
    with fitz.open(pdf_path) as pdf:
        pdf_images = get_pdf_images(pdf)
    
    # Build DOCX tokens
    tokenizer_result = build_docx_tokens(elements)
    docx_tokens, docx_owner, docx_cell_index, docx_is_formula, image_only_cells_map, empty_cells_map, table_structure_map = tokenizer_result
    
    if log_file:
        log_file.write(f"Total DOCX tokens: {len(docx_tokens)}\n")
        log_file.flush()
    sys.stderr.write(f"Total DOCX tokens: {len(docx_tokens)}\n")
    sys.stderr.flush()

    # Build PDF tokens
    pdf_tokens, pdf_bboxes, pdf_pages = build_pdf_tokens(pdf_path)

    if log_file:
        log_file.write(f"Total PDF tokens: {len(pdf_tokens)}\n")
        log_file.flush()
    sys.stderr.write(f"Total PDF tokens: {len(pdf_tokens)}\n")
    sys.stderr.flush()

    # Global alignment
    docx_to_pdf, docx_to_pdf_multi = perform_global_alignment(docx_tokens, pdf_tokens)

    # Group aligned tokens
    element_groups = group_aligned_tokens(
        docx_owner, docx_cell_index, docx_is_formula, 
        docx_to_pdf, docx_to_pdf_multi, 
        pdf_tokens, pdf_bboxes, pdf_pages
    )

    # Filter formula tokens by Y-gap (remove tokens from different formulas)
    filter_formula_tokens_by_y_gap(element_groups)

    # Build element metadata
    metadata_result = build_element_metadata(elements)
    element_texts, element_is_table, element_is_shape_array, element_cell_texts, element_shape_data, element_empty_cells, element_table_structure = metadata_result

    # Pre-compute PDF tokens by page for formula expansion
    from collections import defaultdict
    pdf_on_page = defaultdict(list)
    for idx, (bbox, page_idx) in enumerate(zip(pdf_bboxes, pdf_pages)):
        pdf_on_page[page_idx].append((idx, bbox))

    # Fallback alignment for unaligned shapes
    used_pdf_indices = set()
    for elem_id in element_groups:
        for cell_idx, words in element_groups[elem_id].items():
            for w in words:
                if 'pdf_index' in w:
                    used_pdf_indices.add(w['pdf_index'])
    
    align_unaligned_shapes(
        element_groups, element_is_shape_array, element_shape_data,
        pdf_tokens, pdf_bboxes, pdf_pages, used_pdf_indices
    )

    # Fallback alignment for unaligned table cells (handles out-of-order content)
    # Now uses PyMuPDF cell bbox as primary Y-constraint
    align_unaligned_table_cells(
        element_groups, element_is_table, element_cell_texts,
        pdf_tokens, pdf_bboxes, pdf_pages, used_pdf_indices, 
        pdf_path=pdf_path, log_file=log_file
    )

    # Align images
    image_items, table_cell_images = collect_image_items(elements)
    used_images = set()
    
    if log_file:
        log_file.write(f"\nFound {len(image_items)} standalone images\n")
        log_file.write(f"Found {len(table_cell_images)} table cell images\n")

    # Build final aligned elements
    final_aligned = []
    used_pdf_cells = set()
    
    # Cache for PyMuPDF table detection per page
    pymupdf_table_pages = {}
    
    def has_pymupdf_table_on_page(page_num):
        """Check if PyMuPDF detects any table on this page"""
        if page_num not in pymupdf_table_pages:
            tables = get_table_structure_from_pdf(pdf_path, page_num)
            pymupdf_table_pages[page_num] = len(tables) > 0
        return pymupdf_table_pages[page_num]
    
    # Track tables that need Y-filter (borderless tables)
    borderless_tables = set()

    for elem_id, cell_groups in element_groups.items():
        is_table = element_is_table.get(elem_id, False)
        is_shape_array = element_is_shape_array.get(elem_id, False)
        
        # Check if this is a borderless table (PyMuPDF can't detect it)
        if is_table:
            # Get main page from cell_groups
            main_page = None
            for cell_idx, words in cell_groups.items():
                if words:
                    main_page = words[0]['page']
                    break
            
            # If PyMuPDF doesn't detect table, mark for Y-filter post-processing
            if main_page is not None and not has_pymupdf_table_on_page(main_page):
                borderless_tables.add(elem_id)
                if log_file:
                    log_file.write(f"Borderless table {elem_id} on page {main_page+1}: will apply Y-filter\n")
        
        if is_table:
            # Build table container
            container_result = build_table_container(cell_groups, 0)
            if container_result:
                container_bbox, main_page = container_result
                final_aligned.append({
                    "text": f"Table {elem_id}",
                    "matched_text": "",
                    "bbox": container_bbox,
                    "bboxes": [],
                    "page": main_page,
                    "element_id": elem_id,
                    "confidence": 1.0,
                    "before_align_bboxes": [],
                    "is_table_container": True,
                })
            
            # Build table cells (skip gap detection for now)
            empty_cells_data = element_empty_cells.get(elem_id, [])
            table_cells = build_table_cells(
                elem_id, cell_groups, element_cell_texts, pdf_path, used_pdf_cells, 
                log_file, image_only_cells_map, skip_gap_detection=True, 
                table_cell_images=table_cell_images, empty_cells_data=empty_cells_data
            )
            
            # POST-PROCESS: Apply Y-filter for borderless tables
            # Remove tokens that are too far from the median Y position
            if elem_id in borderless_tables:
                for cell in table_cells:
                    words = cell.get('words', [])
                    if len(words) > 2:
                        # Calculate median Y
                        y_positions = [w['bbox']['y0'] for w in words]
                        y_positions.sort()
                        median_y = y_positions[len(y_positions) // 2]
                        
                        # Estimate max height based on text length (~20px per line, ~80 chars per line)
                        text_len = len(cell.get('text', ''))
                        estimated_lines = max(1, text_len / 80)
                        max_height = estimated_lines * 25 + 30  # Extra margin
                        
                        # Filter words within reasonable Y range
                        filtered_words = [w for w in words 
                                          if abs(w['bbox']['y0'] - median_y) <= max_height]
                        
                        if len(filtered_words) < len(words):
                            if log_file:
                                log_file.write(f"Y-filter {cell.get('element_id')}: {len(words)} -> {len(filtered_words)} words (max_h={max_height:.0f})\n")
                            
                            cell['words'] = filtered_words
                            
                            # Recalculate bbox from filtered words
                            if filtered_words:
                                x0 = min(w['bbox']['x0'] for w in filtered_words)
                                y0 = min(w['bbox']['y0'] for w in filtered_words)
                                x1 = max(w['bbox']['x1'] for w in filtered_words)
                                y1 = max(w['bbox']['y1'] for w in filtered_words)
                                cell['bbox'] = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
            
            # ESTIMATE IMAGE-ONLY CELL POSITIONS BEFORE ALIGNMENT
            num_cols = get_table_columns_from_docx(elem_id, elements)
            if num_cols > 0:
                estimate_image_only_cell_positions(table_cells, num_cols, log_file)
            
            # Apply column alignment
            align_table_columns_x(table_cells, elem_id)
            
            final_aligned.extend(table_cells)
        elif is_shape_array:
            shape_elements = build_shape_elements(elem_id, cell_groups, element_texts, element_shape_data)
            final_aligned.extend(shape_elements)
        else:
            # Non-table elements (including borderless tables now treated as paragraphs)
            non_table_elements = build_non_table_elements(elem_id, cell_groups, element_texts, elements, pdf_on_page)
            final_aligned.extend(non_table_elements)
    
    aligned_cell_images = align_table_cell_images(
        final_aligned, table_cell_images, pdf_images, used_images, log_file
    )
    
    aligned_standalone = align_standalone_images(
        image_items, final_aligned, pdf_images, used_images, elements
    )
    
    if log_file:
        log_file.write(f"Aligned {aligned_cell_images} table cell images\n")
        log_file.write(f"Aligned {aligned_standalone} standalone images\n")

    # Phase 5: Apply gap detection after image alignment (per table)
    from .table_gap_detector import detect_and_remove_gap_cells
    from collections import defaultdict
    
    # Group table cells by parent_element_id
    cells_by_table = defaultdict(list)
    for w in final_aligned:
        if w.get('is_table_cell'):
            parent_id = w.get('parent_element_id')
            if parent_id:
                cells_by_table[parent_id].append(w)
    
    # Apply gap detection per table
    cells_to_remove = []
    for parent_id, table_cells in cells_by_table.items():
        before_count = len(table_cells)
        detect_and_remove_gap_cells(table_cells, log_file)
        after_count = len(table_cells)
        
        # Track removed cells
        if after_count < before_count:
            all_cells = cells_by_table[parent_id]
            for cell in all_cells:
                if cell not in table_cells:
                    cells_to_remove.append(cell)
    
    # Remove filtered cells from final_aligned
    for cell in cells_to_remove:
        if cell in final_aligned:
            final_aligned.remove(cell)

    # Add unaligned elements fallback
    aligned_parent_ids = set()
    for aligned in final_aligned:
        elem_id = aligned.get('element_id')
        parent_id = aligned.get('parent_element_id')
        if parent_id:
            aligned_parent_ids.add(parent_id)
        elif isinstance(elem_id, int):
            aligned_parent_ids.add(elem_id)
        elif isinstance(elem_id, str) and '_page_' in elem_id:
            # Extract base ID from "123_page_10" format
            base_id = int(elem_id.split('_page_')[0])
            aligned_parent_ids.add(base_id)
    
    add_unaligned_elements_fallback(final_aligned, elements, element_texts, aligned_parent_ids, log_file)

    # Collect all used PDF indices
    all_used_indices = set(used_pdf_indices)
    
    # Also mark tokens inside aligned table cells as used (spatially)
    # Group tokens by page for efficiency
    tokens_by_page = defaultdict(list)
    for i, page_idx in enumerate(pdf_pages):
        tokens_by_page[page_idx].append(i)
        
    for item in final_aligned:
        # Check if item has explicit pdf_index
        # Note: final_aligned items have list of bboxes (merged segments)
        # We need to check words/tokens inside
        pass 
        
    # More efficient: Iterate over final_aligned, if it's a table cell, mark tokens in its bbox
    for item in final_aligned:
        if item.get('is_table_cell'):
            page = item.get('page')
            bbox = item.get('bbox')
            if page in tokens_by_page and bbox:
                # Check tokens on this page
                x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
                for token_idx in tokens_by_page[page]:
                    if token_idx in all_used_indices:
                        continue
                        
                    tb = pdf_bboxes[token_idx]
                    # Check overlap/containment
                    # Token bbox: tb[0], tb[1], tb[2], tb[3]
                    # We consider a token used if its center is inside the cell
                    tx_center = (tb[0] + tb[2]) / 2
                    ty_center = (tb[1] + tb[3]) / 2
                    
                    if x0 <= tx_center <= x1 and y0 <= ty_center <= y1:
                        all_used_indices.add(token_idx)

    # Phase 4: Global container cleanup
    cleanup_container_bboxes(final_aligned, log_file)

    if log_file:
        log_file.write(f"\nAligned elements: {len(final_aligned)}\n")
        log_file.write("=== END TRACE LOG ===\n\n")
        log_file.flush()
    sys.stderr.write(f"\nAligned elements: {len(final_aligned)}\n")
    sys.stderr.write("=== END TRACE LOG ===\n\n")
    sys.stderr.flush()

    return build_alignment_result(final_aligned, pdf_tokens, pdf_bboxes, pdf_pages, all_used_indices)
