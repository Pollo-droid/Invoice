import os
import uuid
from fastapi import UploadFile, HTTPException
from pdf2image import convert_from_bytes
import cv2
from API.Backend.config import TEMP_IMAGE_DIR
from API.Backend.file_utils import handle_file_upload

async def convert_pdf_to_png(pdf_file: UploadFile) -> list[str]:
    """Convert PDF to PNG and return a list of paths of the saved PNG images."""
    # Convert PDF to a list of images (pages)
    images = convert_from_bytes(await pdf_file.read())

    if len(images) > 0:
        image_paths = []
        for idx, image in enumerate(images):
            # Save each page as PNG
            unique_filename = f"{uuid.uuid4()}_page_{idx + 1}.png"
            image_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths
    raise HTTPException(status_code=400, detail="Failed to convert PDF to PNG")


async def handle_file_conversion(file: UploadFile) -> list[str]:
    """
    Checks the file type, converts a PDF to PNGs if necessary, or returns the path to the PNG file(s).
    Raises an error if the file type is unsupported.
    """
    # Extract the file extension
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        # Convert PDF to PNG(s)
        return await convert_pdf_to_png(file)

    elif file_extension == 'png':
        # Handle PNG directly
        cv2_img = await handle_file_upload(file)  # Assuming this function processes the image
        unique_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)
        cv2.imwrite(image_path, cv2_img)
        return [image_path]  # Return a list for consistency

    else:
        # Unsupported file format
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and PNG are allowed.")
