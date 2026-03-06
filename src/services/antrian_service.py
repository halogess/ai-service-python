"""
Antrian (Queue) Service
Handles processing tasks from the antrian table
"""

import os
import json
import logging
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from models import Antrian, Dokumen, Bab
from services.pdf_extraction_service import PDFExtractor

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
        
    def get_next_extraction_task(self) -> Optional[Antrian]:
        """
        Get the next task in queue with status 'in_queue' for extraction.
        
        Returns:
            Antrian object or None if queue is empty
        """
        task = self.db.query(Antrian).filter(
            Antrian.antrian_extraction_status == 'in_queue'
        ).order_by(Antrian.antrian_created_at).first()
        
        if task:
            logger.info(f"Found extraction task in queue: ID {task.antrian_id}")
        return task

    def get_next_labeling_task(self) -> Optional[Antrian]:
        """
        Get the next labeling task in queue.
        """
        task = self.db.query(Antrian).filter(
            Antrian.antrian_labeling_status == 'in_queue'
        ).order_by(Antrian.antrian_created_at).first()
        
        if task:
            logger.info(f"Found labeling task in queue: ID {task.antrian_id}")
        return task

    
    def update_extraction_status(self, task: Antrian, status: str, error_message: str = None):
        """
        Update task status (Extraction).
        """
        task.antrian_extraction_status = status
        if error_message:
            task.antrian_error_message = error_message[:255]
        elif status == 'completed':
            task.antrian_error_message = None
        self.db.commit()
        logger.info(f"Task {task.antrian_id} extraction status updated to: {status}")

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
    
    def get_alignment_directory(self, task: Antrian) -> str:
        """
        Get alignment visualization directory (same level as PDF).
        
        Args:
            task: Antrian object
            
        Returns:
            Alignment directory path
        """
        pdf_path = self.get_pdf_path(task)
        pdf_dir = os.path.dirname(pdf_path)
        
        alignment_dir = os.path.join(pdf_dir, 'alignment')
        full_alignment_dir = os.path.join(STORAGE_BASE, alignment_dir)
        
        os.makedirs(full_alignment_dir, exist_ok=True)
        return full_alignment_dir
    
    def process_char_groups(self, task: Antrian) -> dict:
        """
        Process PDF and extract character groups.
        
        Args:
            task: Antrian object
            
        Returns:
            dict with extraction results
        """
        pdf_path = self.get_full_pdf_path(task)
        output_dir = self.get_output_directory(task)
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        with PDFExtractor(pdf_path) as extractor:
            # Extract char groups from all pages
            char_groups = extractor.extract_all_char_groups()
            
            # Save char groups to JSON
            output_path = os.path.join(output_dir, "char_groups.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(char_groups, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved char groups to: {output_path}")
            
            return {
                "pdf_path": pdf_path,
                "output_dir": output_dir,
                "char_groups_file": output_path,
                "page_count": extractor.page_count,
                "total_groups": sum(len(p["groups"]) for p in char_groups),
            }
