from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import User
from database import Base

# Create tables (jika belum ada)
# Base.metadata.create_all(bind=engine)

def create_user(name: str, email: str):
    db = SessionLocal()
    try:
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def get_users():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return users
    finally:
        db.close()

def get_user_by_id(user_id: int):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return user
    finally:
        db.close()

if __name__ == "__main__":
    # Contoh penggunaan
    print("Testing database connection...")
    
    # Buat user baru
    new_user = create_user("John Doe", "john@example.com")
    print(f"Created user: {new_user.name} - {new_user.email}")
    
    # Ambil semua users
    users = get_users()
    print(f"Total users: {len(users)}")
    
    for user in users:
        print(f"ID: {user.id}, Name: {user.name}, Email: {user.email}")