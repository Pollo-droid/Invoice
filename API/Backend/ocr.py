import easyocr
import cv2

def extract_text(cv2_img, numbered_boxes):
    reader = easyocr.Reader(['fr', 'es', 'en'])
    extracted_texts = {}
    for number, bbox in numbered_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped_image = cv2_img[y_min:y_max, x_min:x_max]
        ocr_results = reader.readtext(cropped_image)
        box_text = [{"text": text, "confidence": prob} for (_, text, prob) in ocr_results]
        extracted_texts[number] = box_text
    return extracted_texts
