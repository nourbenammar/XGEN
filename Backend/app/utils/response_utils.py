def success_response(data, message="Success", status_code=200):
    return {
        'success': True,
        'message': message,
        'data': data
    }, status_code

def error_response(message, status_code=400, errors=None):
    response = {
        'success': False,
        'message': message
    }
    if errors:
        response['errors'] = errors
    return response, status_code