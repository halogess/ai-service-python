from sqlalchemy import Column, Integer, BigInteger, String, Text
from database import Base


class DokumenNote(Base):
    __tablename__ = "dokumen_note"

    dnote_id = Column(Integer, primary_key=True, autoincrement=True)
    dnote_ref_tipe = Column(String(16), nullable=False)
    dnote_ref_id = Column(Integer, nullable=False)
    delemen_id = Column(BigInteger, nullable=True)
    dnote_kind = Column(String(10), nullable=False)
    dnote_type = Column(String(30), nullable=True)
    dnote_number = Column(Integer, nullable=True)
    dnote_json_tree = Column(Text, nullable=True)
