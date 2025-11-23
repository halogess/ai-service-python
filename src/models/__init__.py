from database import Base
from .antrian import Antrian
from .buku import Buku
from .bab import Bab
from .dokumen import Dokumen

__all__ = ["Base", "Antrian", "Buku", "Bab", "Dokumen"]