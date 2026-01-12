"""
Reset stuck tasks dari processing ke in_queue
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Override for localhost
os.environ['DB_HOST'] = 'localhost'

from database import SessionLocal
from models import Antrian

db = SessionLocal()

# Find stuck tasks
stuck = db.query(Antrian).filter(
    Antrian.antrian_visual_status == 'processing'
).all()

print(f"Found {len(stuck)} stuck tasks in 'processing' status")

for task in stuck:
    print(f"\nTask ID {task.antrian_id}:")
    print(f"  Tipe: {task.antrian_tipe}")
    print(f"  Dokumen ID: {task.dokumen_id}")
    print(f"  Bab ID: {task.bab_id}")
    print(f"  Created: {task.antrian_created_at}")
    
    # Reset to in_queue
    task.antrian_visual_status = 'in_queue'
    task.antrian_error_message = None

db.commit()
print(f"\n[OK] Reset {len(stuck)} tasks to 'in_queue'")
db.close()
