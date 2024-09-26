from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import pandas as pd
import io
import json
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store uploaded dataset
data_storage = {
    'data': None,  # This will hold the parsed CSV data as a DataFrame
}

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: dict  # Expect a dictionary, not a string

# Endpoint to interact with OpenAI API
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    if data_storage['data'] is None:
        return QueryResponse(response={"error": "Please upload a dataset first."})

    df = data_storage['data']
    columns = df.columns.tolist()

    sample_values = df.head(5).to_dict(orient="records")

    try:
        # Construct the prompt with specific context for generating Vega-Lite spec
        prompt = f"""
        The dataset contains the following columns: {columns}.
        Here are some sample rows from the dataset: {sample_values}.
        The user's query is: {request.prompt}.
        Based on this, generate a valid Vega-Lite JSON specification for a chart.
        Please ensure the chart uses the correct data types (categorical, numerical) and captures the user's intent accurately.
        """
        logging.info("Prompt sent to OpenAI: %s", prompt)

        # Call OpenAI API
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data visualization assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the response text
        response_text = chat_completion.choices[0].message.content

        # Log the raw response to see what's returned by OpenAI
        logging.info("Raw response from OpenAI: %s", response_text)

        # Check if the response is empty
        if not response_text.strip():
            logging.error("OpenAI returned an empty response.")
            raise HTTPException(status_code=500, detail="OpenAI returned an empty response.")

        # Try to parse the response text as JSON
        try:
            vega_spec = json.loads(response_text)  # Ensure it's a valid JSON object
        except json.JSONDecodeError as e:
            logging.error("Failed to parse Vega-Lite JSON: %s", str(e))
            logging.error("OpenAI response was: %s", response_text)
            raise HTTPException(status_code=500, detail="Failed to parse Vega-Lite specification")

        # Return the parsed Vega-Lite JSON as a response
        return QueryResponse(response=vega_spec)

    except Exception as e:
        logging.error("Error generating chart: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate a response")

# Endpoint for uploading the CSV file
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))  # Read the CSV into a Pandas DataFrame
        data_storage['data'] = df
        return {"message": "CSV file uploaded successfully!", "columns": df.columns.tolist(), "sample_data": df.head().to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process the CSV file.")

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running!"}
