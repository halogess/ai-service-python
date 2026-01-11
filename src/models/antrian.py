from sqlalchemy import Column, Integer, String, Enum, DateTime, Index, CheckConstraint
from sqlalchemy.sql import func
from database import Base

class Antrian(Base):
    __tablename__ = "antrian"
    
    antrian_id = Column(Integer, primary_key=True, autoincrement=True)
    antrian_tipe = Column(Enum('dokumen', 'buku'), nullable=False)
    buku_id = Column(Integer, nullable=True)
    bab_id = Column(Integer, nullable=True)
    dokumen_id = Column(Integer, nullable=True)
    antrian_worker = Column(Enum('convert_pdf', 'struktur', 'visual'), nullable=False)
    antrian_convert_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_visual_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_struktur_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_error_message = Column(String(255), nullable=True)
    antrian_created_at = Column(DateTime, server_default=func.current_timestamp())
    antrian_updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    __table_args__ = (
        CheckConstraint(
            "(antrian_tipe = 'buku' AND buku_id IS NOT NULL AND dokumen_id IS NULL) OR "
            "(antrian_tipe = 'dokumen' AND dokumen_id IS NOT NULL AND buku_id IS NULL)",
            name='check_tipe_id_consistency'
        ),
        Index('idx_tipe_buku', 'antrian_tipe', 'buku_id'),
        Index('idx_tipe_dokumen', 'antrian_tipe', 'dokumen_id'),
        Index('idx_created_at', 'antrian_created_at'),
    )