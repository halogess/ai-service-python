from sqlalchemy import Column, Integer, String, BigInteger, Enum, DateTime
from sqlalchemy.sql import func
from database import Base

class Dokumen(Base):
    __tablename__ = "dokumen"
    
    dokumen_id = Column(Integer, primary_key=True, autoincrement=True)
    mhs_nrp = Column(String(9), nullable=False)
    dokumen_tipe = Column(Enum('awal', 'isi', 'akhir', 'lampiran'), nullable=True)
    dokumen_filename = Column(String(255), nullable=False)
    dokumen_filesize_bytes = Column(BigInteger, nullable=True)
    dokumen_status = Column(Enum('dibatalkan', 'dalam_antrian', 'diproses', 'lolos', 'tidak_lolos'), default='dalam_antrian')
    dokumen_skor = Column(Integer, nullable=True)
    dokumen_jumlah_kesalahan = Column(Integer, nullable=True)
    dokumen_docx_path = Column(String(255), nullable=True)
    dokumen_pdf_path = Column(String(255), nullable=True)
    dokumen_images_path = Column(String(255), nullable=True)
    dokumen_created_at = Column(DateTime, server_default=func.current_timestamp())
    dokumen_updated_at = Column(DateTime, server_default=func.current_timestamp())
