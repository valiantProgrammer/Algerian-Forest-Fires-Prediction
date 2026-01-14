from flask import Blueprint, jsonify

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def index():
    """Home route."""
    return jsonify({
        'message': 'Welcome to Flask API',
        'version': '1.0.0'
    })

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })
