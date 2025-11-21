from sqlalchemy import Column, Integer, String, Enum
from db import Base

class Dokumen(Base):
    __tablename__ = "dokumen"
    
    dokumen_id = Column(Integer, primary_key=True)
    mhs_nrp = Column(String(50))
    dokumen_filename = Column(String(255))
    dokumen_status = Column(Enum('dalam_antrian', 'diproses', 'lolos', 'tidak_lolos', 'dibatalkan'))
    dokumen_skor = Column(Integer, nullable=True)
    dokumen_jumlah_kesalahan = Column(Integer, nullable=True)
    dokumen_docx_path = Column(String(500), nullable=True)
    dokumen_pdf_path = Column(String(500), nullable=True)