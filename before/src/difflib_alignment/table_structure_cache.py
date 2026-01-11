"""Cache PyMuPDF table structure for spatial cell alignment"""

import fitz


def get_table_structure_from_pdf(pdf_path, page_num):
    """Extract table structure from PDF page using PyMuPDF.
    
    Returns:
        list of dict: List of tables with cell coordinates indexed by (row, col)
    """
    tables_info = []
    
    try:
        with fitz.open(pdf_path) as doc:
            if page_num >= len(doc):
                return tables_info
                
            page = doc[page_num]
            
            # Use PyMuPDF's table detection
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for tab_idx, tab in enumerate(tabs.tables):
                        table_info = {
                            'table_idx': tab_idx,
                            'bbox': tab.bbox,  # (x0, y0, x1, y1)
                            'row_count': tab.row_count,
                            'col_count': tab.col_count,
                            'cells': [],
                            'cells_by_pos': {}  # (row, col) -> bbox
                        }
                        
                        # Extract cell data using extract() method
                        try:
                            cell_data = tab.extract()
                            if cell_data:
                                for row_idx, row in enumerate(cell_data):
                                    if row:
                                        for col_idx, cell_text in enumerate(row):
                                            # Get cell rect using cells property
                                            cell_rect = None
                                            if hasattr(tab, 'cells') and tab.cells:
                                                # cells is list of Rect objects
                                                cell_index = row_idx * tab.col_count + col_idx
                                                if cell_index < len(tab.cells):
                                                    cr = tab.cells[cell_index]
                                                    cell_rect = (cr.x0, cr.y0, cr.x1, cr.y1)
                                            
                                            if cell_rect:
                                                table_info['cells'].append({
                                                    'row': row_idx,
                                                    'col': col_idx,
                                                    'bbox': cell_rect,
                                                    'text': cell_text if cell_text else ''
                                                })
                                                table_info['cells_by_pos'][(row_idx, col_idx)] = cell_rect
                        except Exception:
                            pass
                        
                        tables_info.append(table_info)
            except Exception:
                # Fall back if table detection fails
                pass
                
    except Exception:
        pass
    
    return tables_info


def get_empty_cell_bbox(pdf_tables, row_idx, col_idx):
    """Find bbox for empty cell by row/col position in PDF table structure.
    
    Args:
        pdf_tables: List of table structures from get_table_structure_from_pdf
        row_idx: Row index of the empty cell
        col_idx: Column index of the empty cell
        
    Returns:
        dict or None: bbox dict {'x0': ..., 'y0': ..., 'x1': ..., 'y1': ...} or None
    """
    for table in pdf_tables:
        cell_bbox = table['cells_by_pos'].get((row_idx, col_idx))
        if cell_bbox:
            return {
                'x0': cell_bbox[0],
                'y0': cell_bbox[1],
                'x1': cell_bbox[2],
                'y1': cell_bbox[3]
            }
    
    return None


class TableStructureCache:
    """Cache for PDF table structures per page"""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self._cache = {}  # page_num -> list of table structures
    
    def get_page_tables(self, page_num):
        """Get cached table structure for page, or extract and cache if not present"""
        if page_num not in self._cache:
            self._cache[page_num] = get_table_structure_from_pdf(self.pdf_path, page_num)
        return self._cache[page_num]
    
    def get_empty_cell_bbox(self, page_num, row_idx, col_idx):
        """Get bbox for empty cell at given position"""
        pdf_tables = self.get_page_tables(page_num)
        return get_empty_cell_bbox(pdf_tables, row_idx, col_idx)
