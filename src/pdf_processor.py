import os
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Convert PDF to images using PyMuPDF
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_document = fitz.open(pdf_path)
    
    image_paths = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image_path = os.path.join(output_dir, f"{filename}-page-{page_num + 1}.jpg")
        pix.save(image_path)
        image_paths.append(image_path)
    
    pdf_document.close()
    return image_paths