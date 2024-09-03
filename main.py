from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles 
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Define request and response models
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

# Endpoint to interact with OpenAI API
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    try:
        logging.info("Received request with prompt: %s", request.prompt)

        # Construct the prompt with specific context for Boston College Tech Support
        tech_support_prompt = f"""
        You are a tech support assistant for Boston College. Your job is to help students and staff with their technology-related issues.
        The user might ask about software, hardware, network issues, or specific Boston College systems like the BC Portal or Agora.
        Always be polite, professional, and provide clear, concise, and actionable advice.

        User's query: {request.prompt}
        """

        # Call the OpenAI API with the custom prompt
        chat_completion = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable tech support assistant for Boston College."
            },
            {
                "role": "user",
                "content": tech_support_prompt,
            }
        ])

        response_text = chat_completion.choices[0].message.content
        formatted_response = response_text.replace("1. ", "\n1. ").replace("2. ", "\n2. ").replace("3. ", "\n3. ").replace("4. ", "\n4. ").replace("5. ", "\n5. ")
        logging.info("Generated response: %s", formatted_response)

        return QueryResponse(response=formatted_response)
    except Exception as e:
        logging.error("Error generating response: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate a response")

# Root endpoint
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

