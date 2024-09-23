from ultralytics import YOLO

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("API/Backend/best.pt")
    return _model
