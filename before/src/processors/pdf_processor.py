import fitz
import os

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        page_info = {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "words": [],
            "blocks": []
        }
        
        words_data = page.get_text("words")
        for word_tuple in words_data:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
            page_info["words"].append({
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "block_no": block_no
            })
        
        text_data = page.get_text("dict")
        for block in text_data.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                page_info["blocks"].append({
                    "bbox": block["bbox"],
                    "text": block_text.strip()
                })
            elif block.get("type") == 1:  # Image block
                bbox = block["bbox"]
                # Add placeholder word for image
                page_info["words"].append({
                    "text": "[IMAGE]",
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "block_no": -1,
                    "is_image": True
                })
        
        pages_data.append(page_info)
    
    pdf_document.close()
    return pages_data

def convert_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_document = fitz.open(pdf_path)
    
    image_paths = []
    scale = 300/72  # DPI scale factor
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        image_path = os.path.join(output_dir, f"{filename}-page-{page_num + 1}.jpg")
        pix.save(image_path)
        image_paths.append(image_path)
    
    pdf_document.close()
    return image_paths
