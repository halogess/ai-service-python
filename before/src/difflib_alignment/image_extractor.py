"""PDF image extraction utilities"""


def get_pdf_images(pdf_doc):
    """Ekstrak semantic images dan vector drawings dari PDF"""
    pdf_images = {}
    
    for page_idx in range(pdf_doc.page_count):
        page = pdf_doc[page_idx]
        images_on_page = []
        
        # Standard Images (Raster)
        try:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                rects = page.get_image_rects(xref)
                for r in rects:
                    images_on_page.append({
                        'bbox': [r.x0, r.y0, r.x1, r.y1],
                        'xref': xref,
                        'type': 'image'
                    })
        except Exception as e:
            print(f"Error getting images on page {page_idx}: {e}")
            
        # Vector Drawings (clustered)
        try:
            drawings = page.get_drawings()
            if drawings and len(drawings) > 20:
                rects = []
                for d in drawings:
                    r = d['rect']
                    w, h = r.width, r.height
                    if w < 2 and h < 2: continue
                    if w > 500 and h > 800: continue
                    if w > 50 and h < 5: continue
                    if h > 50 and w < 5: continue
                    
                    rects.append([r.x0, r.y0, r.x1, r.y1])
                
                clusters = []
                THRESHOLD = 15.0
                
                while rects:
                    current_cluster = rects.pop(0)
                    changed = True
                    
                    while changed:
                        changed = False
                        i = 0
                        while i < len(rects):
                            r = rects[i]
                            c_x0, c_y0, c_x1, c_y1 = current_cluster
                            r_x0, r_y0, r_x1, r_y1 = r
                            
                            gap_x = max(0, r_x0 - c_x1, c_x0 - r_x1)
                            gap_y = max(0, r_y0 - c_y1, c_y0 - r_y1)
                            
                            if gap_x <= THRESHOLD and gap_y <= THRESHOLD:
                                current_cluster[0] = min(c_x0, r_x0)
                                current_cluster[1] = min(c_y0, r_y0)
                                current_cluster[2] = max(c_x1, r_x1)
                                current_cluster[3] = max(c_y1, r_y1)
                                
                                rects.pop(i)
                                changed = True
                            else:
                                i += 1
                    
                    c_w = current_cluster[2] - current_cluster[0]
                    c_h = current_cluster[3] - current_cluster[1]
                    
                    if c_w > 20 and c_h > 20:
                        clusters.append(current_cluster)
                
                for i, c in enumerate(clusters):
                     dummy_xref = -(page_idx * 10000 + i + 5000)
                     images_on_page.append({
                        'bbox': c,
                        'xref': dummy_xref,
                        'type': 'vector'
                     })
                     
        except Exception as e:
            print(f"Error getting drawings on page {page_idx}: {e}")

        if images_on_page:
            pdf_images[page_idx] = images_on_page
            
    return pdf_images
