import json
import streamlit as st
import openai
import csv
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Set API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read CSV and store it as a dictionary or DataFrame
def get_load_csv():
    df = pd.read_csv("/Users/victorlimouzi/code/Pollo-droid/le_invoice/API/Backend/data/Dataset-copro.csv")
    return df

# Updated function to pass invoice text and CSV data to the LLM
def call_llm(invoice_text, csv_data):
    try:
        # Prepare the CSV data as a reference string
        reference_data = ""
        for index, row in csv_data.iterrows():
            reference_data += f"{row['libelleCopro']} - {row['adresse']} - {row['codePostal']} - {row['ville']}\n"

        # Use the reference data in the LLM prompt
        client = openai.OpenAI(os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here is some reference data for matching:\n{reference_data}\n\n"
                        f"Now, extract and structure the following invoice data: {invoice_text}\n"
                        f"The invoice data I sent to you comes from a donut model which tends to have the most reliable number\n"
                        f"The first two texts batches are from another ocr model which holds all the contextual information and some numbers which are less reliable but can be right\n"
                        f"Use your logic to figure which data is best and return it to me. If some value does not look right then do not return it in the json."
                    )
                }
            ],
            functions=[
                {
                    "name": "structure_invoice",
                    "description": "Extract and structure invoice data into a standardized format",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {"type": "string", "description": "The name of the company issuing the invoice"},
                            "invoice_number": {"type": "string", "description": "The unique identifier for the invoice. Usually called N°facture. It can contain letters"},
                            "date": {"type": "string", "description": "The date of the invoice (ISO 8601 format preferred)"},
                            "due_date": {"type": "string", "description": "The due date for payment (ISO 8601 format preferred)"},
                            "total_amount": {"type": "number", "description": "The total amount due on the invoice"},
                            "net_Amount": {"type": "number", "description": "The net amount due on the invoice"},
                            "tax_Amount": {"type": "number", "description": "The tax amount due on the invoice"},
                            "currency": {"type": "string", "description": "The currency used in the invoice (e.g., USD, EUR)"},
                            "billing_address": {"type": "string", "description": "The billing address on the invoice"},
                            "condominium association": {"type": "string", "description": "The name of the condominium association, known in french as syndicat de copropriété, SDC or copropriété for short. Sometimes this name comes after the word REF"},
                            "contract_number": {"type": "number", "description": "The contract number on the invoice called N°contrat"},
                            "SIRET_number": {"type": "number", "description": "A unique identification number assigned to businesses and establishments in France (e.g.,123 456 789 00012)"},
                            "line_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "quantity": {"type": "number"},
                                        "unit_price": {"type": "number"},
                                        "total": {"type": "number"}
                                    }
                                },
                                "description": "An array of items listed on the invoice"
                            }
                        },
                        "required": ["company_name", "invoice_number", "date", "total_amount"]
                    }
                }
            ],
            function_call={"name": "structure_invoice"}
        )

        print("Full API Response:", response)

        if response.choices and response.choices[0].message.function_call:
            try:
                result = json.loads(response.choices[0].message.function_call.arguments)
                print("LLM Result Type:", type(result))
                print("LLM Result Content:", result)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("The response was not in valid JSON format. Here's the raw text:")
                print(response.choices[0].message.function_call.arguments)
                return None
        else:
            print("No function call in the response")
            return None

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None
