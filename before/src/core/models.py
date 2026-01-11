from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class TestingDokumen(db.Model):
    __tablename__ = 'testing_dokumen'
    testing_dokumen_id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    testing_dokumen_nama = db.Column(db.String(255), nullable=False)
    testing_dokumen_path = db.Column(db.String(255), nullable=False)
    testing_dokumen_total_pages = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<TestingDokumen {self.testing_dokumen_nama}>'

class TestingHistory(db.Model):
    __tablename__ = 'testing_history'
    testing_history_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    testing_dokumen_id = db.Column(db.Integer, nullable=False)
    testing_history_description = db.Column(db.String(255))
    testing_history_processing_time = db.Column(db.Float)
    testing_history_created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

    def __repr__(self):
        return f'<TestingHistory {self.testing_history_id}>'

class TestingGroundTruth(db.Model):
    __tablename__ = 'testing_ground_truth'
    testing_ground_truth_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    testing_dokumen_id = db.Column(db.Integer, nullable=False)
    testing_ground_truth_page = db.Column(db.Integer, nullable=False)
    testing_ground_truth_bbox = db.Column(db.JSON, nullable=False)
    testing_ground_truth_label = db.Column(db.String(50), nullable=False)
    testing_ground_truth_word = db.Column(db.String(255))
    testing_ground_truth_confidence = db.Column(db.Float, default=1.0)
    testing_ground_truth_created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

    def __repr__(self):
        return f'<TestingGroundTruth {self.testing_ground_truth_id}>'

class TestingPrediction(db.Model):
    __tablename__ = 'testing_prediction'
    testing_prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    testing_history_id = db.Column(db.Integer, nullable=False)
    testing_prediction_page = db.Column(db.Integer, nullable=False)
    testing_prediction_bbox_x0 = db.Column(db.Float)
    testing_prediction_bbox_y0 = db.Column(db.Float)
    testing_prediction_bbox_x1 = db.Column(db.Float)
    testing_prediction_bbox_y1 = db.Column(db.Float)
    testing_prediction_label = db.Column(db.String(50), nullable=False)
    testing_prediction_word = db.Column(db.Text)
    testing_prediction_confidence = db.Column(db.Float, default=1.0)
    testing_prediction_created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    
    @property
    def testing_prediction_bbox(self):
        return [self.testing_prediction_bbox_x0, self.testing_prediction_bbox_y0, 
                self.testing_prediction_bbox_x1, self.testing_prediction_bbox_y1]

    def __repr__(self):
        return f'<TestingPrediction {self.testing_prediction_id}>'

class DokumenElemen(db.Model):
    __tablename__ = 'dokumen_elemen'
    delemen_id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True, autoincrement=True, nullable=False)
    dpart_id = db.Column(db.Integer, nullable=True)
    delemen_sequence = db.Column(db.Integer, nullable=True)
    delemen_type = db.Column(db.String(100), nullable=True)
    delemen_json_tree = db.Column(db.JSON, nullable=True)
    delemen_xml = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<DokumenElemen {self.delemen_id}>'

class DokumenElemenVisual(db.Model):
    __tablename__ = 'dokumen_elemen_visual'
    dev_id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True, autoincrement=True, nullable=False)
    dokumen_id = db.Column(db.Integer, nullable=True)
    dev_bbox_x0 = db.Column(db.Float, nullable=True)
    dev_bbox_y0 = db.Column(db.Float, nullable=True)
    dev_bbox_x1 = db.Column(db.Float, nullable=True)
    dev_bbox_y1 = db.Column(db.Float, nullable=True)
    dev_page = db.Column(db.Integer, nullable=True)
    dev_label = db.Column(db.String(50), nullable=True)
    dev_text = db.Column(db.Text, nullable=True)
    dokumen_elemen_id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), nullable=True)

    def __repr__(self):
        return f'<DokumenElemenVisual {self.dev_id}>'


class DokumenPart(db.Model):
    __tablename__ = 'dokumen_part'
    dpart_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    dsec_id = db.Column(db.Integer, nullable=False)
    dpart_type = db.Column(db.String(20), nullable=False)
    dpart_position = db.Column(db.String(10), nullable=True)

    def __repr__(self):
        return f'<DokumenPart {self.dpart_id}>'

class DokumenSection(db.Model):
    __tablename__ = 'dokumen_section'
    dsec_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    dokumen_id = db.Column(db.Integer, nullable=True)
    dsec_index = db.Column(db.Integer, nullable=True)
    dsec_type = db.Column(db.String(50), nullable=True)
    dsec_has_title_page = db.Column(db.Boolean, nullable=False, default=False)
    dsec_different_odd_even = db.Column(db.Boolean, nullable=False, default=False)
    dsec_page_num_format = db.Column(db.String(32), nullable=True)
    dsec_page_num_start = db.Column(db.Integer, nullable=True)
    dsec_page_width_twips = db.Column(db.Integer, nullable=True)
    dsec_page_height_twips = db.Column(db.Integer, nullable=True)
    dsec_orientation = db.Column(db.String(20), nullable=True)
    dsec_margin_top_twips = db.Column(db.Integer, nullable=True)
    dsec_margin_bottom_twips = db.Column(db.Integer, nullable=True)
    dsec_margin_left_twips = db.Column(db.Integer, nullable=True)
    dsec_margin_right_twips = db.Column(db.Integer, nullable=True)
    dsec_header_margin_twips = db.Column(db.Integer, nullable=True)
    dsec_footer_margin_twips = db.Column(db.Integer, nullable=True)
    dsec_gutter_twips = db.Column(db.Integer, nullable=True)
    dsec_gutter_position = db.Column(db.String(20), nullable=True)
    dsec_column_count = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<DokumenSection {self.dsec_id}>'
