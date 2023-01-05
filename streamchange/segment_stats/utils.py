
def has_method(obj, method_name):
    method = getattr(obj, method_name, None)
    return callable(method)