import os
import fitz  # PyMuPDF

def extract_layout_data_from_pdf(pdf_path):
    """
    Mengekstrak teks, gambar, dan blok dari setiap halaman PDF.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    doc = fitz.open(pdf_path)
    all_pages_data = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")
        page_width, page_height = page.rect.width, page.rect.height
        
        page_data = {
            "width": page_width,
            "height": page_height,
            "words": [],
            "blocks": [],
            "images": []
        }

        page_blocks = sorted(blocks["blocks"], key=lambda b: (b['bbox'][1], b['bbox'][0]))
        block_idx = 0
        text_blocks = 0
        image_blocks = 0

        for b in page_blocks:
            block_bbox = b["bbox"]

            if b["type"] == 0:
                text_blocks += 1
                block_text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                        for word in span.get("text", "").split():
                            if word.strip():
                                page_data["words"].append({
                                    "text": word,
                                    "bbox": span["bbox"],
                                    "block_no": block_idx
                                })
                
                page_data["blocks"].append({
                    "text": block_text.strip(),
                    "bbox": block_bbox,
                    "type": "text"
                })
                block_idx += 1

            elif b["type"] == 1:
                image_blocks += 1
                page_data["images"].append({
                    "bbox": block_bbox,
                    "block_no": block_idx
                })
                page_data["blocks"].append({
                    "text": "",
                    "bbox": block_bbox,
                    "type": "image"
                })
                block_idx += 1
        
        logger.info(f"Page {page_num+1}: {text_blocks} text blocks, {image_blocks} image blocks from get_text()")
        
        image_list = page.get_images()
        logger.info(f"Page {page_num+1}: {len(image_list)} images from get_images()")
        
        if image_list and not page_data["images"]:
            logger.info(f"Page {page_num+1}: Using fallback method to extract images")
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        for rect in img_rects:
                            bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                            page_data["images"].append({
                                "bbox": bbox,
                                "block_no": block_idx
                            })
                            page_data["blocks"].append({
                                "text": "",
                                "bbox": bbox,
                                "type": "image"
                            })
                            logger.info(f"  Found image at bbox: {bbox}")
                            block_idx += 1
                except Exception as e:
                    logger.warning(f"  Failed to extract image {img_idx}: {e}")
        
        logger.info(f"Page {page_num+1}: Total {len(page_data['images'])} images detected")

        all_pages_data.append(page_data)
        
    doc.close()
    return all_pages_data

def extract_text_from_pdf(pdf_path):
    """
    Extract text and bounding boxes from PDF per word
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        list: Text data per page with word-level bounding boxes
    """
    pdf_document = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        page_info = {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "words": [],
            "blocks": []  # Keep for visualization
        }
        
        # Extract words with actual bboxes
        words_data = page.get_text("words")  # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        for word_tuple in words_data:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
            page_info["words"].append({
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "block_no": block_no
            })
        
        # Extract blocks for visualization
        text_data = page.get_text("dict")
        for block in text_data.get("blocks", []):
            if block.get("type") == 0:
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                page_info["blocks"].append({
                    "bbox": block["bbox"],
                    "text": block_text.strip()
                })
        
        pages_data.append(page_info)
    
    pdf_document.close()
    return pages_data

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Convert PDF to images using PyMuPDF
    
    Args:
        pdf_path: Path ke PDF file (e.g., /app/storage/dokumen/1/2/docx/file.pdf)
        output_dir: Directory output untuk images (e.g., /app/storage/dokumen/1/2/images)
    
    Returns:
        List of image paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    
    image_paths = []
    for page_num in range(len(pdf_document)):
        # Get page
        page = pdf_document[page_num]
        
        # Convert to image (300 DPI for good quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Save image: nama-file-page-1.jpg
        image_path = os.path.join(output_dir, f"{filename}-page-{page_num + 1}.jpg")
        pix.save(image_path)
        image_paths.append(image_path)
    
    pdf_document.close()
    return image_paths

def get_pdf_path_from_docx(docx_path):
    """
    Convert docx path to pdf path
    
    Args:
        docx_path: dokumen/1/2/docx/Bab 3 - Desain Sistem.docx
    
    Returns:
        dokumen/1/2/pdf/Bab 3 - Desain Sistem.pdf
    """
    return docx_path.replace('/docx/', '/pdf/').replace('.docx', '.pdf')

def get_images_dir_from_docx(docx_path):
    """
    Get images directory from docx path
    
    Args:
        docx_path: dokumen/1/2/docx/Bab 3 - Desain Sistem.docx
    
    Returns:
        dokumen/1/2/images
    """
    parts = docx_path.split('/')
    # dokumen/{id_orang}/{id_dokumen}/images
    return '/'.join(parts[:3]) + '/images'