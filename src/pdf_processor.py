import os
import math
import fitz  # PyMuPDF

def detect_table_regions_from_drawings(page):
    """
    Deteksi semua cluster garis vektor di halaman sebagai kandidat region border/tabel.
    Tidak ada nilai minimal, jadi semua garis dan cluster dikembalikan apa adanya.
    """
    line_boxes = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    for d in drawings:
        for item in d.get("items", []):
            if not item or item[0] != "l":
                continue

            p1 = item[1]
            p2 = item[2]
            x0, y0 = p1
            x1, y1 = p2

            if x0 == x1 and y0 == y1:
                continue

            bx0 = min(x0, x1)
            by0 = min(y0, y1)
            bx1 = max(x0, x1)
            by1 = max(y0, y1)

            line_boxes.append([bx0, by0, bx1, by1])

    if not line_boxes:
        return []

    clusters = []
    for b in line_boxes:
        placed = False
        for idx, cb in enumerate(clusters):
            if not (b[2] < cb[0] or cb[2] < b[0] or b[3] < cb[1] or cb[3] < b[1]):
                clusters[idx] = [
                    min(cb[0], b[0]),
                    min(cb[1], b[1]),
                    max(cb[2], b[2]),
                    max(cb[3], b[3]),
                ]
                placed = True
                break
        if not placed:
            clusters.append(b[:])

    return clusters


def extract_layout_data_from_pdf(pdf_path):
    """
    Mengekstrak teks, gambar, dan blok dari setiap halaman PDF.
    - blocks: per blok (text / image) dari get_text("dict")
    - words: per kata dari get_text("words") dengan bbox per kata
             dan block_no yang disesuaikan dengan index di page_data["blocks"]
    """
    import logging
    logger = logging.getLogger(__name__)
    
    doc = fitz.open(pdf_path)
    all_pages_data = []

    for page_num, page in enumerate(doc):
        blocks_dict = page.get_text("dict")
        page_width, page_height = page.rect.width, page.rect.height

        page_data = {
            "width": page_width,
            "height": page_height,
            "words": [],
            "blocks": [],
            "images": [],
        }

        # ----- 1) Bangun blocks (text & image) seperti biasa -----
        # Sort blocks supaya urutan rapi top-to-bottom, left-to-right
        page_blocks = sorted(
            blocks_dict["blocks"],
            key=lambda b: (b["bbox"][1], b["bbox"][0])
        )

        block_idx = 0
        text_blocks = 0
        image_blocks = 0

        # Kita simpan daftar text-block untuk mapping words -> block_no
        text_block_infos = []  # list of {"idx": block_idx, "bbox": [...]}

        for b in page_blocks:
            block_bbox = b["bbox"]

            if b["type"] == 0:  # text block
                text_blocks += 1
                block_text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "

                page_data["blocks"].append({
                    "text": block_text.strip(),
                    "bbox": block_bbox,
                    "type": "text",
                })

                text_block_infos.append({
                    "idx": block_idx,
                    "bbox": block_bbox,
                })

                block_idx += 1

            elif b["type"] == 1:  # image block
                image_blocks += 1

                page_data["images"].append({
                    "bbox": block_bbox,
                    "block_no": block_idx,
                })
                page_data["blocks"].append({
                    "text": "",
                    "bbox": block_bbox,
                    "type": "image",
                })

                block_idx += 1

        logger.info(
            f"Page {page_num+1}: {text_blocks} text blocks, {image_blocks} image blocks from get_text(dict)"
        )

        # ----- 2) Isi words[] pakai get_text("words") (per kata) -----
        # words_data: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words_data = page.get_text("words")

        def find_block_idx_for_word(x0, y0, x1, y1):
            """
            Cari text-block yang menampung pusat kata (cx, cy).
            Kalau tidak ada yang benar-benar menampung, ambil yang secara vertikal paling dekat.
            """
            if not text_block_infos:
                return -1

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            # Kandidat yang benar-benar mengandung pusat kata
            candidates = []
            for tb in text_block_infos:
                bx0, by0, bx1, by1 = tb["bbox"]
                if (bx0 - 1) <= cx <= (bx1 + 1) and (by0 - 1) <= cy <= (by1 + 1):
                    candidates.append(tb)

            if candidates:
                # Ambil yang paling "kecil" atau pertama saja
                return candidates[0]["idx"]

            # Fallback: pilih text-block dengan jarak vertikal minimal
            best = None
            best_dist = None
            for tb in text_block_infos:
                bx0, by0, bx1, by1 = tb["bbox"]
                byc = (by0 + by1) / 2.0
                dist = abs(byc - cy)
                if best is None or dist < best_dist:
                    best = tb
                    best_dist = dist

            return best["idx"] if best is not None else -1

        for w in words_data:
            x0, y0, x1, y1, text, _block_no_orig, _line_no, _word_no = w
            text = text.strip()
            if not text:
                continue

            block_no = find_block_idx_for_word(x0, y0, x1, y1)

            page_data["words"].append({
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "block_no": block_no,
            })

        # ----- 3) Deteksi image fallback dari get_images (seperti kode lama) -----
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
                                "block_no": block_idx,
                            })
                            page_data["blocks"].append({
                                "text": "",
                                "bbox": bbox,
                                "type": "image",
                            })
                            logger.info(f"  Found image at bbox: {bbox}")
                            block_idx += 1
                except Exception as e:
                    logger.warning(f"  Failed to extract image {img_idx}: {e}")

        logger.info(f"Page {page_num+1}: Total {len(page_data['images'])} images detected")

        # ----- 4) Deteksi table border dari vektor (punya kamu) -----
        try:
            table_regions_border = detect_table_regions_from_drawings(page)
            logger.info(
                f"Page {page_num+1}: Detected {len(table_regions_border)} table regions from borders"
            )
            page_data["table_regions_border"] = table_regions_border
        except Exception as e:
            logger.warning(f"Page {page_num+1}: Failed to detect table borders: {e}")
            page_data["table_regions_border"] = []

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