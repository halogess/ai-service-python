from .document_routes import document_bp
from .history_routes import history_bp
from .analysis_routes import analysis_bp
from .ground_truth_routes import ground_truth_bp
from .pdf_routes import pdf_bp
from .pymupdf_routes import pymupdf_bp
from .dokumen_elemen_routes import dokumen_elemen_bp
from .classification_routes import classification_bp

def register_routes(app):
    app.register_blueprint(document_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(ground_truth_bp)
    app.register_blueprint(pdf_bp)
    app.register_blueprint(pymupdf_bp)
    app.register_blueprint(dokumen_elemen_bp)
    app.register_blueprint(classification_bp)

