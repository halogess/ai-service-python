from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.sql import func
from database import Base


class Aturan(Base):
    __tablename__ = "aturan"

    aturan_id = Column(Integer, primary_key=True, autoincrement=True)
    aturan_versi = Column(String(255), nullable=False)
    aturan_status = Column(
        Enum('diproses', 'menunggu_review', 'tidak_aktif', 'aktif', 'gagal'),
        nullable=False,
        default='tidak_aktif'
    )
    aturan_template_file_path = Column(String(255), nullable=True)
    aturan_template_pdf_path = Column(String(255), nullable=True)
    aturan_created_at = Column(DateTime, server_default=func.current_timestamp())
    aturan_updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
