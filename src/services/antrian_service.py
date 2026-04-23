"""
Antrian (Queue) Service
Handles processing tasks from the antrian table
"""

import os
import logging
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy import or_
from sqlalchemy.orm import Session

from models import Antrian, Buku, Dokumen, Bab, Aturan

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("STORAGE_BASE", "/app/storage")


class AntrianService:
    """Service class for processing antrian (queue) tasks"""
    
    def __init__(self, db: Session):
        """
        Initialize antrian service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        
    def get_next_labeling_task(self) -> Optional[Antrian]:
        """
        Get the next labeling task in queue.
        """
        in_queue_query = self.db.query(Antrian).filter(
            Antrian.antrian_labeling_status == 'in_queue'
        )

        active_buku_ids_subquery = self.db.query(Antrian.buku_id).filter(
            Antrian.antrian_tipe == 'buku',
            Antrian.buku_id.isnot(None),
            or_(
                Antrian.antrian_labeling_status.in_(['processing', 'completed']),
                Antrian.antrian_extraction_status == 'completed'
            )
        ).distinct().subquery()

        task = in_queue_query.filter(
            Antrian.antrian_tipe == 'buku',
            Antrian.buku_id.in_(active_buku_ids_subquery)
        ).order_by(Antrian.antrian_created_at).first()

        if task is None:
            task = in_queue_query.filter(
                Antrian.antrian_tipe == 'aturan'
            ).order_by(Antrian.antrian_created_at).first()

        if task is None:
            task = in_queue_query.order_by(Antrian.antrian_created_at).first()
        
        if task:
            logger.info(f"Found labeling task in queue: ID {task.antrian_id}")
        return task

    
    def update_labeling_status(self, task: Antrian, status: str, error_message: str = None):
        """
        Update task status (Labeling).
        """
        task.antrian_labeling_status = status
        if error_message:
            task.antrian_error_message = error_message[:255]
        elif status == 'completed':
            task.antrian_error_message = None
        self.db.commit()
        logger.info(f"Task {task.antrian_id} labeling status updated to: {status}")

    def update_validation_status(self, task: Antrian, status: str, error_message: str = None):
        """
        Update task status (Validation).
        """
        task.antrian_validation_status = status
        if error_message:
            task.antrian_error_message = error_message[:255]
        elif status == 'completed':
            task.antrian_error_message = None
        self.db.commit()
        logger.info(f"Task {task.antrian_id} validation status updated to: {status}")

    def is_task_cancelled(self, task: Antrian) -> bool:
        """Return True when the queue target has been cancelled by the user."""
        if not task:
            return False

        if task.antrian_tipe == 'dokumen':
            if not task.dokumen_id:
                return False
            dokumen = self.db.query(Dokumen).filter(
                Dokumen.dokumen_id == task.dokumen_id
            ).first()
            return bool(dokumen and dokumen.dokumen_status == 'dibatalkan')

        if task.antrian_tipe == 'buku':
            buku_id = task.buku_id
            if not buku_id and task.bab_id:
                bab = self.db.query(Bab).filter(
                    Bab.bab_id == task.bab_id
                ).first()
                buku_id = bab.buku_id if bab else None

            if not buku_id:
                return False

            buku = self.db.query(Buku).filter(
                Buku.buku_id == buku_id
            ).first()
            return bool(buku and buku.buku_status == 'dibatalkan')

        return False

    def mark_task_cancelled(self, task: Antrian, error_message: str = "Dibatalkan oleh pengguna."):
        """Clear active stages for a cancelled task so workers stop picking it up."""
        if not task:
            return

        changed = False
        for field_name in (
            'antrian_extraction_status',
            'antrian_labeling_status',
            'antrian_validation_status',
        ):
            current_value = getattr(task, field_name, None)
            if current_value in ('in_queue', 'processing'):
                setattr(task, field_name, None)
                changed = True

        if not changed:
            return

        task.antrian_error_message = (error_message or "Dibatalkan oleh pengguna.")[:255]
        task.antrian_updated_at = datetime.utcnow()
        self.db.commit()
        logger.info(f"Task {task.antrian_id} cleared because the resource was cancelled")

    def get_task_reference(self, task: Antrian) -> Tuple[str, int]:
        """
        Resolve logical reference target for the task.

        Returns:
            (ref_tipe, ref_id)
            - dokumen -> (dokumen, dokumen_id)
            - buku (task type) -> (bab, bab_id)
        """
        if task.antrian_tipe == 'dokumen':
            if not task.dokumen_id:
                raise ValueError(f"Task {task.antrian_id} has no dokumen_id")
            return 'dokumen', task.dokumen_id

        if task.antrian_tipe == 'buku':
            if not task.bab_id:
                raise ValueError(f"Task {task.antrian_id} has no bab_id")
            return 'bab', task.bab_id

        if task.antrian_tipe == 'aturan':
            if not task.aturan_id:
                raise ValueError(f"Task {task.antrian_id} has no aturan_id")
            return 'aturan', task.aturan_id

        raise ValueError(f"Unknown antrian_tipe: {task.antrian_tipe}")

        
    def get_pdf_path(self, task: Antrian) -> str:
        """
        Get PDF path based on task type (dokumen or buku/bab).
        
        Args:
            task: Antrian object
            
        Returns:
            PDF file path
            
        Raises:
            ValueError: If PDF path not found
        """
        if task.antrian_tipe == 'dokumen':
            dokumen = self.db.query(Dokumen).filter(
                Dokumen.dokumen_id == task.dokumen_id
            ).first()
            if not dokumen or not dokumen.dokumen_pdf_path:
                raise ValueError(f"PDF path not found for dokumen_id: {task.dokumen_id}")
            return dokumen.dokumen_pdf_path
            
        elif task.antrian_tipe == 'buku':
            bab = self.db.query(Bab).filter(
                Bab.bab_id == task.bab_id
            ).first()
            if not bab or not bab.bab_pdf_path:
                raise ValueError(f"PDF path not found for bab_id: {task.bab_id}")
            return bab.bab_pdf_path

        elif task.antrian_tipe == 'aturan':
            aturan = self.db.query(Aturan).filter(
                Aturan.aturan_id == task.aturan_id
            ).first()
            if not aturan or not aturan.aturan_template_pdf_path:
                raise ValueError(f"PDF path not found for aturan_id: {task.aturan_id}")
            return aturan.aturan_template_pdf_path
            
        else:
            raise ValueError(f"Unknown antrian_tipe: {task.antrian_tipe}")
    
    def get_full_pdf_path(self, task: Antrian) -> str:
        """
        Get full absolute PDF path.
        
        Args:
            task: Antrian object
            
        Returns:
            Full absolute path to PDF file
        """
        relative_path = self.get_pdf_path(task)
        return os.path.join(STORAGE_BASE, relative_path)
    
    def get_output_directory(self, task: Antrian) -> str:
        """
        Get output directory for processed files.
        
        Args:
            task: Antrian object
            
        Returns:
            Output directory path
        """
        pdf_path = self.get_pdf_path(task)
        pdf_dir = os.path.dirname(pdf_path)
        base_dir = os.path.dirname(pdf_dir)
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        output_dir = os.path.join(base_dir, 'extraction', pdf_filename)
        full_output_dir = os.path.join(STORAGE_BASE, output_dir)
        
        os.makedirs(full_output_dir, exist_ok=True)
        return full_output_dir
    
