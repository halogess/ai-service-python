"""
Check database readiness for alignment pipeline.
Flow: antrian -> dokumen/bab -> dokumen_section -> dokumen_part -> dokumen_elemen
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _connect_with_fallback():
    print('Trying to connect to database...')
    print('\nAttempt 1: Using .env config (host.docker.internal)')

    try:
        from database import SessionLocal
        from models import Antrian

        db = SessionLocal()
        count = db.query(Antrian).count()
        db.close()
        print(f'[OK] Connected! Found {count} antrian records')
        return SessionLocal
    except Exception as e:
        print(f'[FAILED] {str(e)[:100]}')
        print('\nAttempt 2: Trying localhost instead...')

    os.environ['DB_HOST'] = 'localhost'

    try:
        import importlib
        import database
        import models

        importlib.reload(database)
        importlib.reload(models)

        from database import SessionLocal
        from models import Antrian

        db = SessionLocal()
        count = db.query(Antrian).count()
        db.close()
        print(f'[OK] Connected with localhost! Found {count} antrian records')
        return SessionLocal
    except Exception as e2:
        print(f'[FAILED] {str(e2)[:100]}')
        print('\n[ERROR] Cannot connect to database!')
        print('\nPlease check:')
        print('  1. Is Docker running? (docker ps)')
        print('  2. Is MySQL container running?')
        print('  3. Try: docker-compose up -d')
        return None


def check_database():
    session_factory = _connect_with_fallback()
    if not session_factory:
        return

    from models import Antrian, Bab, Dokumen, DokumenSection, DokumenPart, DokumenElemen

    def resolve_ref(task):
        if task.antrian_tipe == 'dokumen':
            return 'dokumen', task.dokumen_id
        if task.antrian_tipe == 'buku':
            return 'bab', task.bab_id
        return task.antrian_tipe, None

    def build_ref_tipe_filter(column, ref_tipe):
        if ref_tipe == 'bab':
            return column.in_(('bab', 'buku'))
        return column == ref_tipe

    db = session_factory()
    try:
        print('=' * 80)
        print('DATABASE CHECK: ANTRIAN -> REF -> ELEMEN')
        print('=' * 80)

        antrian_count = db.query(Antrian).count()
        print(f'\n[1] ANTRIAN: {antrian_count} records')

        if antrian_count > 0:
            antrian = db.query(Antrian).order_by(Antrian.antrian_id.desc()).first()
            if antrian:
                print('    Sample task:')
                print(f'      ID: {antrian.antrian_id}')
                print(f'      Tipe: {antrian.antrian_tipe}')
                print(f'      Dokumen ID: {antrian.dokumen_id}')
                print(f'      Bab ID: {antrian.bab_id}')
                print(f'      Extraction Status: {antrian.antrian_extraction_status}')
                print(f'      Labeling Status: {antrian.antrian_labeling_status}')
                print(f'      Validation Status: {antrian.antrian_validation_status}')

                ref_tipe, ref_id = resolve_ref(antrian)

                if ref_tipe == 'dokumen' and ref_id:
                    doc = db.query(Dokumen).get(ref_id)
                    if doc:
                        print('\n[2] DOKUMEN: Found')
                        print(f'      ID: {doc.dokumen_id}')
                        print(f'      Filename: {doc.dokumen_filename}')
                        print(f'      PDF: {doc.dokumen_pdf_path}')
                    else:
                        print(f'\n[2] DOKUMEN: NOT FOUND for ID {ref_id}')
                elif ref_tipe == 'bab' and ref_id:
                    bab = db.query(Bab).get(ref_id)
                    if bab:
                        print('\n[2] BAB: Found')
                        print(f'      ID: {bab.bab_id}')
                        print(f'      Filename: {bab.bab_filename}')
                        print(f'      PDF: {bab.bab_pdf_path}')
                    else:
                        print(f'\n[2] BAB: NOT FOUND for ID {ref_id}')
                else:
                    print('\n[2] REFERENCE: reference id is NULL')

                if ref_id:
                    sections = db.query(DokumenSection).filter(
                        build_ref_tipe_filter(DokumenSection.dsec_ref_tipe, ref_tipe),
                        DokumenSection.dsec_ref_id == ref_id
                    ).all()
                    print(f'\n[3] DOKUMEN_SECTION ({ref_tipe}:{ref_id}): {len(sections)} records')

                    if sections:
                        sec = sections[0]
                        print('      Sample section:')
                        print(f'        ID: {sec.dsec_id}')
                        print(f'        Index: {sec.dsec_index}')
                        print(f'        Page size: {sec.dsec_page_width_twips}x{sec.dsec_page_height_twips} twips')

                        parts = db.query(DokumenPart).filter_by(dsec_id=sec.dsec_id).all()
                        body_parts = [p for p in parts if p.dpart_type == 'body']
                        print(f'\n[4] DOKUMEN_PART: {len(parts)} records')
                        print(f'      Body parts: {len(body_parts)}')

                        if body_parts:
                            part = body_parts[0]
                            elements = db.query(DokumenElemen).filter_by(dpart_id=part.dpart_id).all()
                            print(f'\n[5] DOKUMEN_ELEMEN: {len(elements)} records for sample body part')

                            all_part_ids = [p.dpart_id for p in body_parts]
                            total_elements = db.query(DokumenElemen).filter(
                                DokumenElemen.dpart_id.in_(all_part_ids)
                            ).count()
                            print(f'      Total elements (all body parts): {total_elements}')

        print('\n' + '=' * 80)
        print('ALIGNMENT READINESS CHECK')
        print('=' * 80)

        labeling_tasks = db.query(Antrian).filter(
            Antrian.antrian_labeling_status == 'in_queue'
        ).all()
        print(f"\nLabeling tasks (antrian_labeling_status='in_queue'): {len(labeling_tasks)}")

        processing = db.query(Antrian).filter(Antrian.antrian_labeling_status == 'processing').count()
        completed = db.query(Antrian).filter(Antrian.antrian_labeling_status == 'completed').count()
        failed = db.query(Antrian).filter(Antrian.antrian_labeling_status == 'failed').count()

        print(f'  - in_queue: {len(labeling_tasks)}')
        print(f'  - processing: {processing}')
        print(f'  - completed: {completed}')
        print(f'  - failed: {failed}')

        for task in labeling_tasks[:5]:
            print(f'\n  Task ID {task.antrian_id}:')
            print(f'    Tipe: {task.antrian_tipe}')
            print(f'    Dokumen ID: {task.dokumen_id}')
            print(f'    Bab ID: {task.bab_id}')
            print(f'    Labeling Status: {task.antrian_labeling_status}')

            ref_tipe, ref_id = resolve_ref(task)

            if not ref_id:
                missing_ref = 'dokumen_id' if ref_tipe == 'dokumen' else 'bab_id'
                print(f'    [!] CANNOT ALIGN - NO {missing_ref.upper()}!')
                continue

            elem_count = db.query(DokumenElemen).join(
                DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id
            ).join(
                DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id
            ).filter(
                build_ref_tipe_filter(DokumenSection.dsec_ref_tipe, ref_tipe),
                DokumenSection.dsec_ref_id == ref_id,
                DokumenPart.dpart_type == 'body'
            ).count()

            print(f'    Elements: {elem_count}')
            if elem_count == 0:
                print('    [!] CANNOT ALIGN - NO ELEMENTS!')
            else:
                print(f'    [OK] Ready for alignment ({elem_count} elements)')

        print('\n' + '=' * 80)

    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == '__main__':
    check_database()
