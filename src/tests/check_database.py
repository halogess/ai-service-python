"""
Cek data lengkap: antrian → dokumen → dokumen_section → dokumen_part → dokumen_elemen
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def check_database():
    # Try different connection methods
    print("Trying to connect to database...")
    print("\nAttempt 1: Using .env config (host.docker.internal)")
    
    try:
        from database import SessionLocal
        from models import Antrian, Dokumen, DokumenSection, DokumenPart, DokumenElemen
        
        db = SessionLocal()
        count = db.query(Antrian).count()
        db.close()
        print(f"[OK] Connected! Found {count} antrian records")
        use_default = True
    except Exception as e:
        print(f"[FAILED] {str(e)[:100]}")
        print("\nAttempt 2: Trying localhost instead...")
        
        # Override environment
        os.environ['DB_HOST'] = 'localhost'
        
        try:
            # Reload modules to use new env
            import importlib
            import database
            import models
            importlib.reload(database)
            importlib.reload(models)
            
            from database import SessionLocal
            from models import Antrian, Dokumen, DokumenSection, DokumenPart, DokumenElemen
            
            db = SessionLocal()
            count = db.query(Antrian).count()
            db.close()
            print(f"[OK] Connected with localhost! Found {count} antrian records")
            use_default = False
        except Exception as e2:
            print(f"[FAILED] {str(e2)[:100]}")
            print("\n[ERROR] Cannot connect to database!")
            print("\nPlease check:")
            print("  1. Is Docker running? (docker ps)")
            print("  2. Is MySQL container running?")
            print("  3. Try: docker-compose up -d")
            return
    
    # If we got here, connection works
    from database import SessionLocal
    from models import Antrian, Dokumen, DokumenSection, DokumenPart, DokumenElemen
    db = SessionLocal()
    try:
        print("="*80)
        print("DATABASE CHECK: ANTRIAN -> DOKUMEN -> ELEMEN")
        print("="*80)
        
        # 1. Check Antrian
        antrian_count = db.query(Antrian).count()
        print(f"\n[1] ANTRIAN: {antrian_count} records")
        
        if antrian_count > 0:
            antrian = db.query(Antrian).filter_by(antrian_tipe='visual').first()
            if antrian:
                print(f"    Sample visual task:")
                print(f"      ID: {antrian.antrian_id}")
                print(f"      Dokumen ID: {antrian.dokumen_id}")
                print(f"      Status: {antrian.antrian_status}")
                print(f"      PDF: {antrian.antrian_pdf_path}")
                
                # 2. Check Dokumen
                if antrian.dokumen_id:
                    doc = db.query(Dokumen).get(antrian.dokumen_id)
                    if doc:
                        print(f"\n[2] DOKUMEN: Found")
                        print(f"      ID: {doc.dokumen_id}")
                        print(f"      Nama: {doc.dokumen_nama}")
                        print(f"      PDF: {doc.dokumen_pdf_path}")
                        
                        # 3. Check Sections
                        sections = db.query(DokumenSection).filter_by(dokumen_id=doc.dokumen_id).all()
                        print(f"\n[3] DOKUMEN_SECTION: {len(sections)} records")
                        if sections:
                            sec = sections[0]
                            print(f"      Sample section:")
                            print(f"        ID: {sec.dsec_id}")
                            print(f"        Index: {sec.dsec_index}")
                            print(f"        Page size: {sec.dsec_page_width_twips}x{sec.dsec_page_height_twips} twips")
                            
                            # 4. Check Parts
                            parts = db.query(DokumenPart).filter_by(dsec_id=sec.dsec_id).all()
                            print(f"\n[4] DOKUMEN_PART: {len(parts)} records")
                            body_parts = [p for p in parts if p.dpart_type == 'body']
                            print(f"      Body parts: {len(body_parts)}")
                            
                            if body_parts:
                                part = body_parts[0]
                                print(f"      Sample body part:")
                                print(f"        ID: {part.dpart_id}")
                                print(f"        Type: {part.dpart_type}")
                                
                                # 5. Check Elements
                                elements = db.query(DokumenElemen).filter_by(dpart_id=part.dpart_id).all()
                                print(f"\n[5] DOKUMEN_ELEMEN: {len(elements)} records for this part")
                                
                                # Count all elements for all body parts
                                all_part_ids = [p.dpart_id for p in body_parts]
                                total_elements = db.query(DokumenElemen).filter(
                                    DokumenElemen.dpart_id.in_(all_part_ids)
                                ).count()
                                print(f"      Total elements (all body parts): {total_elements}")
                                
                                if elements:
                                    print(f"\n      Sample elements:")
                                    for i, elem in enumerate(elements[:3]):
                                        text = elem.delemen_text[:50] if elem.delemen_text else '[no text]'
                                        print(f"        {i+1}. ID={elem.delemen_id} Seq={elem.delemen_sequence} Type={elem.delemen_type}")
                                        print(f"           Text: '{text}'")
                                else:
                                    print(f"      [!] NO ELEMENTS IN THIS PART!")
                            else:
                                print(f"      [!] NO BODY PARTS!")
                        else:
                            print(f"      [!] NO SECTIONS!")
                    else:
                        print(f"\n[2] DOKUMEN: NOT FOUND for ID {antrian.dokumen_id}")
                else:
                    print(f"\n[2] DOKUMEN: antrian.dokumen_id is NULL")
            else:
                print(f"    No visual tasks found")
        
        # Summary check for alignment
        print("\n" + "="*80)
        print("ALIGNMENT READINESS CHECK")
        print("="*80)
        
        # Check by visual status (CORRECT)
        visual_tasks = db.query(Antrian).filter(
            Antrian.antrian_visual_status == 'in_queue'
        ).all()
        print(f"\nVisual tasks (antrian_visual_status='in_queue'): {len(visual_tasks)}")
        
        # Also check other statuses
        processing = db.query(Antrian).filter(
            Antrian.antrian_visual_status == 'processing'
        ).count()
        completed = db.query(Antrian).filter(
            Antrian.antrian_visual_status == 'completed'
        ).count()
        failed = db.query(Antrian).filter(
            Antrian.antrian_visual_status == 'failed'
        ).count()
        
        print(f"  - in_queue: {len(visual_tasks)}")
        print(f"  - processing: {processing}")
        print(f"  - completed: {completed}")
        print(f"  - failed: {failed}")
        
        for task in visual_tasks[:5]:  # Check first 5
            print(f"\n  Task ID {task.antrian_id}:")
            print(f"    Tipe: {task.antrian_tipe}")
            print(f"    Worker: {task.antrian_worker}")
            print(f"    Dokumen ID: {task.dokumen_id}")
            print(f"    Bab ID: {task.bab_id}")
            print(f"    Visual Status: {task.antrian_visual_status}")
            
            if task.dokumen_id:
                # Count elements for this document
                elem_count = db.query(DokumenElemen).join(
                    DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id
                ).join(
                    DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id
                ).filter(
                    DokumenSection.dokumen_id == task.dokumen_id,
                    DokumenPart.dpart_type == 'body'
                ).count()
                
                print(f"    Elements: {elem_count}")
                
                if elem_count == 0:
                    print(f"    [!] CANNOT ALIGN - NO ELEMENTS!")
                else:
                    print(f"    [OK] Ready for alignment ({elem_count} elements)")
            else:
                print(f"    [!] CANNOT ALIGN - NO DOKUMEN_ID!")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    check_database()
