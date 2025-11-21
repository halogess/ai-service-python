from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from models.antrian import Antrian

def ambil_satu_job_visual(session: Session) -> Optional[Antrian]:
    try:
        job = session.query(Antrian).filter(
            and_(
                Antrian.antrian_worker == 'visual',
                Antrian.antrian_visual_status == 'in_queue',
                Antrian.antrian_convert_status == 'completed'
            )
        ).order_by(Antrian.antrian_created_at).with_for_update().first()
        
        if job is None:
            session.rollback()
            return None
        
        job.antrian_visual_status = 'processing'
        session.flush()
        session.commit()
        return job
        
    except Exception as e:
        session.rollback()
        raise e