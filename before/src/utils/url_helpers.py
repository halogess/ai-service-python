"""URL helper to maintain backward compatibility with old endpoint names."""

from flask import url_for as flask_url_for

# Mapping old endpoint names to new blueprint names
ENDPOINT_MAP = {
    'home': 'document.home',
    'upload_pdf': 'document.upload_pdf',
    'document_detail': 'document.document_detail',
    'serve_document_image': 'document.serve_document_image',
    'update_testing_dokumen_id': 'document.update_testing_dokumen_id',
    
    'history_detail': 'history.history_detail',
    'confusion_matrix_page': 'history.confusion_matrix_page',
    'get_confusion_matrix': 'history.get_confusion_matrix',
    'serve_result_image': 'history.serve_result_image',
    'get_annotations': 'history.get_annotations',
    'merge_annotations': 'history.merge_annotations',
    'update_history_desc': 'history.update_history_desc',
    'delete_history': 'history.delete_history',
    
    'get_analysis_progress': 'analysis.get_analysis_progress',
    'analyze': 'analysis.analyze',
    
    'view_ground_truth': 'ground_truth.view_ground_truth',
    'get_ground_truth_data': 'ground_truth.get_ground_truth_data',
    'merge_ground_truth': 'ground_truth.merge_ground_truth',
    'save_ground_truth': 'ground_truth.save_ground_truth',
}

def url_for(endpoint, **values):
    """Wrapper for url_for that handles old endpoint names."""
    if endpoint in ENDPOINT_MAP:
        endpoint = ENDPOINT_MAP[endpoint]
    return flask_url_for(endpoint, **values)
