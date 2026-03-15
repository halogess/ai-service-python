
import os
import fitz
from typing import Dict, Any, List
from docling.document_converter import DocumentConverter
from models import Dokumen
from database import SessionLocal

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")

class DoclingService:
    def __init__(self):
        pass

    def classify_document(self, doc_id: int) -> Dict[str, Any]:
        """
        Run Docling classification on ENTIRE document (all pages at once).
        Matches `api_docling_classify_document` from classification_routes.py.
        """
        db = SessionLocal()
        try:
            doc = db.query(Dokumen).get(doc_id)
            if not doc:
                raise ValueError(f"Document with ID {doc_id} not found")
            pdf_path = os.path.join(STORAGE_BASE, doc.dokumen_pdf_path)
            return self.classify_pdf(pdf_path=pdf_path, ref_tipe='dokumen', ref_id=doc_id)
            
        except Exception as e:
            import traceback
            print(f"[DoclingService] Error: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
        finally:
            db.close()

    def classify_pdf(self, pdf_path: str, ref_tipe: str = 'dokumen', ref_id: int = None) -> Dict[str, Any]:
        """Run Docling classification for a given PDF path."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            ref_label = f"{ref_tipe}:{ref_id}" if ref_id is not None else ref_tipe
            print(f"[Docling] Processing PDF ({ref_label}) path={pdf_path}")

            # Convert PDF with Docling
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            docling_doc = result.document

            # Open PDF to get page dimensions for coordinate conversion
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)

            page_heights = {}
            for page_num in range(total_pages):
                page_heights[page_num + 1] = pdf_doc[page_num].rect.height

            pdf_doc.close()

            predictions_by_page = {str(p): [] for p in range(1, total_pages + 1)}

            # Helper to process items
            def process_item(item, label_override=None):
                provs = getattr(item, 'prov', None)
                if not provs:
                    return

                if not isinstance(provs, list):
                    provs = [provs]

                label = label_override
                if not label:
                    label = str(item.label).split('.')[-1].lower() if hasattr(item, 'label') else 'text'

                text_content = ''
                if label == 'table':
                    text_content = '[Table]'
                elif label == 'picture':
                    text_content = '[Picture]'
                elif label == 'formula':
                    text_content = item.text if hasattr(item, 'text') else '[Formula]'
                else:
                    text_content = item.text if hasattr(item, 'text') else ''

                for prov in provs:
                    if not (hasattr(prov, 'page_no') and hasattr(prov, 'bbox') and prov.bbox):
                        continue

                    page = prov.page_no
                    if page < 1 or page > total_pages:
                        continue

                    bbox = prov.bbox
                    ph = page_heights.get(page, 842)
                    y_top = ph - bbox.t
                    y_bottom = ph - bbox.b

                    predictions_by_page[str(page)].append({
                        'text': text_content,
                        'bbox': [bbox.l, y_top, bbox.r, y_bottom],
                        'label': label,
                        'confidence': 1.0,
                        'source': 'docling'
                    })

            # Extract from texts
            for text in docling_doc.texts:
                process_item(text)

            # Extract from tables
            for table in docling_doc.tables:
                process_item(table, label_override='table')

            # Extract from pictures
            if hasattr(docling_doc, 'pictures'):
                for picture in docling_doc.pictures:
                    process_item(picture, label_override='picture')

            # Extract from formulas
            if hasattr(docling_doc, 'formulas'):
                for formula in docling_doc.formulas:
                    process_item(formula, label_override='formula')

            # Sort predictions
            for page_key in predictions_by_page:
                predictions_by_page[page_key].sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))

            total_preds = sum(len(p) for p in predictions_by_page.values())
            print(f"[DoclingService] {ref_label}: Found {total_preds} predictions")

            return {
                'success': True,
                'total_pages': total_pages,
                'bbox_unit': 'pdf_points',
                'predictions_by_page': predictions_by_page
            }

        except Exception as e:
            import traceback
            print(f"[DoclingService] Error: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
