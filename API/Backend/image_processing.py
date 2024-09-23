import cv2
from API.Backend.ocr import extract_text
from API.Backend.model import get_model
from API.Backend.donut_extraction import donut_extraction
import tempfile

def process_image(cv2_img, yolo_model, donut_model):
    # Perform inference with YOLO model
    results = yolo_model(cv2_img)

    # Extract bounding boxes, classes, and labels
    boxes = results[0].boxes.xyxy.numpy()
    classes = results[0].boxes.cls.numpy()
    names = results[0].names

    # Convert class indices to class names
    labels = [names[int(cls)] for cls in classes]

    # Filter and enumerate bounding boxes for 'Paragraph' and 'Table'
    paragraph_boxes = [(i+1, box) for i, (box, label) in enumerate(zip(boxes, labels)) if label == 'Paragraph']
    table_boxes = [(i+1, box) for i, (box, label) in enumerate(zip(boxes, labels)) if label == 'Table']

    # Draw boxes on the image
    image_with_boxes = draw_boxes(cv2_img, paragraph_boxes, table_boxes)

    # Extract text for paragraphs and tables
    paragraph_texts = extract_text(cv2_img, paragraph_boxes)
    table_texts = extract_text(cv2_img, table_boxes)

# Add Donut extraction
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, cv2_img)
        donut_results = donut_extraction(temp_file.name, donut_model)

    return image_with_boxes, paragraph_texts, table_texts, donut_results

def draw_boxes(cv2_img, paragraph_boxes, table_boxes):
    image_with_boxes = cv2_img.copy()
    for number, box in paragraph_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_with_boxes, f'Paragraph {number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for number, box in table_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f'Table {number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image_with_boxes
