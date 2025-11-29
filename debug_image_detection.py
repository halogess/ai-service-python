import fitz
import sys

def debug_pdf_images(pdf_path):
    """Debug script untuk cek deteksi gambar di PDF"""
    print(f"Opening PDF: {pdf_path}\n")
    
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        print(f"=== PAGE {page_num + 1} ===")
        print(f"Size: {page.rect.width} x {page.rect.height}")
        
        # Method 1: get_text("dict")
        blocks = page.get_text("dict")
        text_blocks = 0
        image_blocks = 0
        
        for b in blocks["blocks"]:
            if b["type"] == 0:
                text_blocks += 1
            elif b["type"] == 1:
                image_blocks += 1
                print(f"  Image block found: bbox={b['bbox']}")
        
        print(f"Text blocks: {text_blocks}, Image blocks: {image_blocks}")
        
        # Method 2: get_images()
        image_list = page.get_images()
        print(f"Images from get_images(): {len(image_list)}")
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            print(f"  Image {img_idx + 1}: xref={xref}")
            
            try:
                rects = page.get_image_rects(xref)
                print(f"    Rectangles: {len(rects)}")
                for rect in rects:
                    print(f"      bbox=[{rect.x0}, {rect.y0}, {rect.x1}, {rect.y1}]")
            except Exception as e:
                print(f"    Error getting rects: {e}")
        
        # Method 3: get_drawings()
        drawings = page.get_drawings()
        print(f"Drawings: {len(drawings)}")
        
        print()
    
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_image_detection.py <pdf_path>")
        sys.exit(1)
    
    debug_pdf_images(sys.argv[1])
