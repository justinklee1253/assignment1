from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import io
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import re
from io import StringIO
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://justinklee1253.github.io/chatbotclient/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data store
data_store = {"df": None}

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: Dict[str, Any]

def execute_python(code: str) -> str:
    """Execute Python code and capture its output."""
    if data_store["df"] is None:
        return "No data loaded. Please upload a dataset first."
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        # Make DataFrame and libraries available
        df = data_store["df"]
        # Clean the code
        code = code.replace('```python', '').replace('```', '').strip()
        # Execute
        exec(code)
        output = mystdout.getvalue().strip()
        sys.stdout = old_stdout
        return output if output else "Analysis completed successfully."
    except Exception as e:
        sys.stdout = old_stdout
        logger.error(f"Python execution error: {str(e)}")
        return f"Error in analysis: {str(e)}"

def create_visualization(spec: Dict) -> Dict:
    """Create a Vega-Lite visualization with the given specification."""
    df = data_store["df"]
    
    try:
        base_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": 300,
            "data": {"values": df.to_dict(orient="records")}
        }
        
        final_spec = {**base_spec, **spec}
        
        if "encoding" in final_spec:
            for axis in ["x", "y"]:
                if axis in final_spec["encoding"]:
                    field = final_spec["encoding"][axis].get("field")
                    if field and field in df.columns:
                        if pd.api.types.is_numeric_dtype(df[field]):
                            final_spec["encoding"][axis]["type"] = "quantitative"
                        else:
                            final_spec["encoding"][axis]["type"] = "nominal"

        return final_spec
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return {"error": str(e)}

@app.post("/query")
async def process_query(request: QueryRequest):
    logger.info(f"Received query: {request.prompt}")
    
    try:
        df = data_store["df"]
        if df is None:
            return QueryResponse(response={"text": "Please upload a dataset first."})
        
        column_info = {col: str(df[col].dtype) for col in df.columns}
        sample_data = df.head(3).to_dict(orient="records")
        
        system_prompt = f"""You are a data analysis assistant. You have access to a dataset with these columns:
{json.dumps(column_info, indent=2)}

Sample data:
{json.dumps(sample_data, indent=2)}

For statistical analysis, use Python code like these examples:

Example 1 - Basic Statistics:
```python
# Range analysis
min_val = df['Weight'].min()
max_val = df['Weight'].max()
print(f"The range is from {{min_val:,.0f}} to {{max_val:,.0f}} lbs")
```

Example 2 - Group Analysis:
```python
# Average by group
result = df.groupby('Origin')['MPG'].mean()
for group, val in result.items():
    print(f"{{group}}: {{val:.1f}} MPG")
```

Example 3 - Conditional Analysis:
```python
# Specific condition
filtered = df[df['Origin'] == 'US']['MPG'].median()
print(f"The median value is {{filtered:.1f}}")
```

For visualizations, use these Vega-Lite specs:

Bar Chart:
{{"mark": "bar",
  "encoding": {{
    "x": {{"field": "Origin"}},
    "y": {{"field": "MPG", "aggregate": "mean"}}
  }}
}}

Scatter Plot:
{{"mark": "point",
  "encoding": {{
    "x": {{"field": "Weight"}},
    "y": {{"field": "MPG"}},
    "color": {{"field": "Origin"}}
  }}
}}

Histogram:
{{"mark": "bar",
  "encoding": {{
    "x": {{"field": "MPG", "bin": true}},
    "y": {{"aggregate": "count"}}
  }}
}}

Guidelines:
1. Write complete code with actual column names
2. Include units in output (MPG, lbs, etc.)
3. Format large numbers with commas
4. Include statistical context
5. Choose appropriate visualizations"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ]

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python",
                        "description": "Execute Python code using pandas DataFrame 'df'",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "create_visualization",
                        "description": "Create a Vega-Lite visualization",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "spec": {
                                    "type": "object",
                                    "properties": {
                                        "mark": {"type": "string"},
                                        "encoding": {"type": "object"}
                                    },
                                    "required": ["mark", "encoding"]
                                }
                            },
                            "required": ["spec"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )

        response = completion.choices[0].message
        logger.info(f"GPT response: {response}")
        
        if response.tool_calls:
            results = []
            viz_spec = None
            
            for tool_call in response.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    
                    if tool_call.function.name == "execute_python":
                        result = execute_python(arguments["code"])
                        if result:
                            results.append(result)
                            
                    elif tool_call.function.name == "create_visualization":
                        viz_spec = create_visualization(arguments["spec"])
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    continue
            
            if viz_spec and not viz_spec.get("error"):
                if results:
                    viz_spec["title"] = "\n".join(results)
                return QueryResponse(response=viz_spec)
            
            if results:
                return QueryResponse(response={"text": "\n".join(results)})
            return QueryResponse(response={"text": "Sorry, I couldn't process that request. Please try rephrasing it."})

        if response.content:
            return QueryResponse(response={"text": response.content})
            
        return QueryResponse(response={"text": "I couldn't understand how to analyze that. Please try rephrasing your question."})

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(response={"text": "An error occurred while processing your request. Please try again."})

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    logger.info(f"Receiving file: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        data_store["df"] = df
        
        response = {
            "message": "File uploaded successfully",
            "columns": df.columns.tolist(),
            "sample": df.head().to_dict(orient="records")
        }
        logger.info(f"File processed successfully: {df.shape}")
        return response
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "running"}