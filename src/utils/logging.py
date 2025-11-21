import logging
import sys
from config import get_settings

def setup_logger():
    settings = get_settings()
    
    logger = logging.getLogger("ai_service")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger