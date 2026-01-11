from sqlalchemy import Column, Integer, String, BigInteger, Text
from database import Base

class DokumenElemen(Base):
    __tablename__ = "dokumen_elemen"
    
    delemen_id = Column(BigInteger, primary_key=True, autoincrement=True)
    dpart_id = Column(Integer, nullable=True)
    delemen_sequence = Column(Integer, nullable=True)
    delemen_type = Column(String(100), nullable=True)
    delemen_json_tree = Column(Text, nullable=True)  # JSON stored as longtext
    delemen_xml = Column(Text, nullable=False)
