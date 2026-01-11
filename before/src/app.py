from flask import Flask
import os
from core.models import db
from routes import register_routes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://jessica:pass123@localhost:3307/db_korektor_buku'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ASSETS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your-secret-key'

os.makedirs(app.config['ASSETS_FOLDER'], exist_ok=True)

db.init_app(app)
register_routes(app)

# Add url_for helper to Jinja2 context
from utils.url_helpers import url_for as custom_url_for
app.jinja_env.globals['url_for'] = custom_url_for

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
