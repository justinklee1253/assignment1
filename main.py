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
    allow_origins=["*"],
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
    
    # Create string buffers for stdout and stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # Store original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect stdout and stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # Make DataFrame and libraries available
        df = data_store["df"]
        # Clean the code
        code = code.replace('```python', '').replace('```', '').strip()
        
        # Execute the code in a new local namespace
        local_vars = {'df': df, 'pd': pd, 'np': np, 'stats': stats}
        exec(code, globals(), local_vars)
        
        # Get the output
        output = stdout_buffer.getvalue().strip()
        error_output = stderr_buffer.getvalue().strip()
        
        # If there's no stdout but there are variables created, show their values
        if not output and not error_output:
            # Check for any new variables created by the code
            for var_name, value in local_vars.items():
                if var_name not in ['df', 'pd', 'np', 'stats'] and not var_name.startswith('_'):
                    output += f"{var_name}: {value}\n"
        
        return output if output else error_output if error_output else "No output generated. Try printing your results."
    
    except Exception as e:
        return f"Error in analysis: {str(e)}"
    
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

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

Example 1 - Average by Group:
```python
# Average MPG by origin
result = df.groupby('Origin')['MPG'].mean()
for origin, mpg in result.items():
    print(f"{{origin}}: {{mpg:.1f}} MPG")
```

Example 2 - Filtered Analysis:
```python
# Cars with high horsepower
high_hp = df[df['Horsepower'] > 150]['MPG'].mean()
print(f"Average MPG for high horsepower cars: {{high_hp:.1f}}")
count = len(df[df['Horsepower'] > 150])
print(f"Number of cars with high horsepower: {{count}}")
```

Example 3 - Correlation Analysis:
```python
# Calculate correlation
correlation = df['MPG'].corr(df['Horsepower'])
print(f"Correlation between MPG and Horsepower: {{correlation:.3f}}")
print("\\nSummary Statistics:")
print(df[['MPG', 'Horsepower']].describe().round(2))
```

For visualizations, use these Vega-Lite specs:

Bar Chart:
{{"mark": "bar",
  "encoding": {{
    "x": {{"field": "Origin"}},
    "y": {{"field": "MPG", "aggregate": "mean"}},
    "tooltip": [
      {{"field": "Origin"}},
      {{"field": "MPG", "aggregate": "mean", "format": ".1f"}}
    ]
  }},
  "title": "Average MPG by Origin"
}}

Scatter Plot:
{{"mark": "point",
  "encoding": {{
    "x": {{"field": "Weight", "title": "Weight (lbs)"}},
    "y": {{"field": "MPG", "title": "Miles per Gallon"}},
    "color": {{"field": "Origin"}},
    "tooltip": [
      {{"field": "Name"}},
      {{"field": "MPG", "format": ".1f"}},
      {{"field": "Weight", "format": ",.0f"}}
    ]
  }},
  "title": "MPG vs Weight by Origin"
}}

Histogram:
{{"mark": "bar",
  "encoding": {{
    "x": {{"field": "MPG", "bin": {{"binned": false}}, "title": "Miles per Gallon"}},
    "y": {{"aggregate": "count", "title": "Number of Cars"}},
    "tooltip": [
      {{"field": "MPG", "bin": {{"binned": false}}, "format": ".1f"}},
      {{"aggregate": "count", "title": "Count"}}
    ]
  }},
  "title": "Distribution of MPG"
}}

Guidelines:
1. Always print results for statistical analysis
2. Include units in output (MPG, lbs, etc.)
3. Format large numbers with commas
4. Round decimals to 1-2 places for readability
5. Include descriptive titles and labels
6. Add tooltips to visualizations when relevant"""

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
