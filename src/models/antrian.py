from sqlalchemy import Column, Integer, String, Enum, DateTime, func
from db import Base

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
    antrian_created_at = Column(DateTime, default=func.current_timestamp())
    antrian_updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())