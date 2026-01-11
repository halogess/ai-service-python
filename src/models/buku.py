from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.sql import func
from database import Base

class Buku(Base):
    __tablename__ = "buku"
    
    buku_id = Column(Integer, primary_key=True, autoincrement=True)
    mhs_nrp = Column(String(9), nullable=False)
    buku_judul = Column(String(255), nullable=False)
    buku_status = Column(Enum('dibatalkan', 'dalam_antrian', 'diproses', 'lolos', 'tidak_lolos'), default='dalam_antrian')
    buku_skor = Column(Integer, nullable=True)
    buku_jumlah_kesalahan = Column(Integer, nullable=True)
    buku_jumlah_bab = Column(Integer, default=0)
    buku_created_at = Column(DateTime, server_default=func.current_timestamp())
    buku_updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())