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
    antrian_extraction_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_labeling_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_validation_status = Column(Enum('in_queue', 'processing', 'completed', 'failed'), nullable=True)
    antrian_error_message = Column(String(255), nullable=True)
    antrian_created_at = Column(DateTime, server_default=func.current_timestamp())
    antrian_updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    __table_args__ = (
        Index('idx_tipe_id', 'antrian_tipe', 'buku_id', 'dokumen_id'),
        Index('idx_created_at', 'antrian_created_at'),
    )