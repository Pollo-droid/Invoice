from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import cv2
from API.Backend.model import get_model
from API.Backend.donut_extraction import load_donut_model
from API.Backend.image_processing import process_image
from API.Backend.file_utils import setup_temp_directory, cleanup_old_images
from API.Backend.llm import call_llm, get_load_csv
from API.Backend.file_conversion import handle_file_conversion

app = FastAPI()

app.state.csv = get_load_csv()
app.state.yolo_model = get_model()
app.state.donut_model = load_donut_model()

# Allow all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup and mount temp directory
setup_temp_directory(app)

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_invoice')
async def receive_files(files: list[UploadFile] = File(...)):
    results = []

    # Process each file in the request
    for file in files:
        try:
            # Call the handle_file_conversion to check the file type and handle conversion
            image_path = await handle_file_conversion(file)

            # Read the image for further processing
            cv2_img = cv2.imread(image_path)

            # Process the image (using YOLO, Donut, etc.)
            image_with_boxes, paragraph_texts, table_texts, donut_results = process_image(cv2_img, app.state.yolo_model, app.state.donut_model)

            # Save the image with bounding boxes (overwriting the old path)
            cv2.imwrite(image_path, image_with_boxes)

            # Data extraction for the response
            data = {
                "paragraphs": paragraph_texts,
                "tables": table_texts,
                "image_url": f"/temp_images/{os.path.basename(image_path)}",
                "donut_extraction": donut_results
            }

            # Combine extracted text
            extracted_text = " ".join([p['text'] for paragraphs in data['paragraphs'].values() for p in paragraphs])
            extracted_tables = " ".join([item.get('text', '') for table in data.get('tables', {}).values() for item in table])
            extracted_donut_extraction = " ".join([f"{key}: {value}" for key, value in data.get('donut_extraction', {}).items()])
            combined_text = f"{extracted_text} {extracted_tables} {extracted_donut_extraction}"

            # Call the LLM with the combined text
            llm_result = call_llm(combined_text, app.state.csv)

            # Prepare the result for this file
            result = {
                "file_name": file.filename,
                "llm_result": llm_result,
                "image_url": f"/temp_images/{os.path.basename(image_path)}"
            }

            # Add the result to the list of results
            results.append(result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")

    # Return the results for all processed files
    return JSONResponse(content={"results": results})

# Cleanup task
app.on_event("startup")(cleanup_old_images)
app.on_event("shutdown")(cleanup_old_images)
