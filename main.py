# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from openai import OpenAI
# import os
# import pandas as pd
# import io
# import json
# from dotenv import load_dotenv
# import logging

# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load OpenAI API key from environment variable
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Store uploaded dataset
# data_storage = {
#     'data': None,  # This will hold the parsed CSV data as a DataFrame
# }

# # Define request and response models
# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: dict  # Expect a dictionary, not a string

# # Endpoint to interact with OpenAI API
# @app.post("/query", response_model=QueryResponse)
# async def query_openai(request: QueryRequest):
#     if data_storage['data'] is None:
#         return QueryResponse(response={"error": "Please upload a dataset first."})

#     df = data_storage['data']
#     columns = df.columns.tolist()

#     sample_values = df.head(5).to_dict(orient="records")

#     try:
#         # Construct the prompt with specific context for generating Vega-Lite spec
#         prompt = f"""
#         The dataset contains the following columns: {columns}.
#         Here are some sample rows from the dataset: {sample_values}.
#         The user's query is: {request.prompt}.
#         Based on this, generate a valid Vega-Lite JSON specification for a chart.
#         Please ensure the chart uses the correct data types (categorical, numerical) and captures the user's intent accurately.
#         """
#         logging.info("Prompt sent to OpenAI: %s", prompt)

#         # Call OpenAI API
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful data visualization assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         # Extract the response text
#         response_text = chat_completion.choices[0].message.content

#         # Log the raw response to see what's returned by OpenAI
#         logging.info("Raw response from OpenAI: %s", response_text)

#         # Check if the response is empty
#         if not response_text.strip():
#             logging.error("OpenAI returned an empty response.")
#             raise HTTPException(status_code=500, detail="OpenAI returned an empty response.")

#         # Try to parse the response text as JSON
#         try:
#             vega_spec = json.loads(response_text)  # Ensure it's a valid JSON object
#         except json.JSONDecodeError as e:
#             logging.error("Failed to parse Vega-Lite JSON: %s", str(e))
#             logging.error("OpenAI response was: %s", response_text)
#             raise HTTPException(status_code=500, detail="Failed to parse Vega-Lite specification")

#         # Return the parsed Vega-Lite JSON as a response
#         return QueryResponse(response=vega_spec)

#     except Exception as e:
#         logging.error("Error generating chart: %s", str(e))
#         raise HTTPException(status_code=500, detail="Failed to generate a response")

# # Endpoint for uploading the CSV file
# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     if file.content_type != 'text/csv':
#         raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

#     content = await file.read()
#     try:
#         df = pd.read_csv(io.BytesIO(content))  # Read the CSV into a Pandas DataFrame
#         data_storage['data'] = df
#         return {"message": "CSV file uploaded successfully!", "columns": df.columns.tolist(), "sample_data": df.head().to_dict(orient="records")}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Failed to process the CSV file.")

# # Root endpoint
# @app.get("/")
# async def read_root():
#     return {"message": "API is running!"}

# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Any
# import pandas as pd
# import io
# import json
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Ensure CORS is properly configured
# origins = [
#     "http://localhost:3000",
#     "http://localhost:5173",  # Vite default port
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     return "nominal"

# def execute_tool(tool_name: str, args: dict) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded. Please upload a CSV file first."

#     try:
#         if tool_name == "analyze_data":
#             column = args.get("column")
#             if column not in df.columns:
#                 return f"Column '{column}' not found in dataset."
            
#             stats = df[column].describe()
#             return f"Analysis of {column}:\n{stats.to_string()}"

#         elif tool_name == "create_visualization":
#             chart_type = args.get("chart_type", "bar")
#             x = args.get("x")
#             y = args.get("y")
            
#             if x not in df.columns or (y and y not in df.columns):
#                 return "Specified columns not found in dataset."
            
#             spec = {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": df.to_dict(orient="records")},
#                 "mark": chart_type,
#                 "encoding": {
#                     "x": {"field": x, "type": get_column_type(df, x)},
#                     "y": {"field": y, "type": get_column_type(df, y)} if y else {"aggregate": "count"}
#                 },
#                 "width": "container",
#                 "height": 300
#             }
#             return json.dumps(spec)

#         return "Unknown tool"
#     except Exception as e:
#         logger.error(f"Tool execution error: {str(e)}")
#         return f"Error executing tool: {str(e)}"

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that helps analyze data and create visualizations. Use these tools:
#              1. analyze_data(column): Get statistical analysis of a column
#              2. create_visualization(chart_type, x, y=None): Create a visualization
             
#              Think step by step:
#              1. Understand what the user wants to analyze or visualize
#              2. Choose the appropriate tool
#              3. Explain your reasoning
#              4. Execute the tool and interpret results"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             functions=[
#                 {
#                     "name": "analyze_data",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "column": {"type": "string", "description": "Column name to analyze"}
#                         },
#                         "required": ["column"]
#                     }
#                 },
#                 {
#                     "name": "create_visualization",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "point"]},
#                             "x": {"type": "string", "description": "Column for x-axis"},
#                             "y": {"type": "string", "description": "Column for y-axis"}
#                         },
#                         "required": ["chart_type", "x"]
#                     }
#                 }
#             ],
#             function_call="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.function_call:
#             function_name = response.function_call.name
#             function_args = json.loads(response.function_call.arguments)
#             result = execute_tool(function_name, function_args)
            
#             try:
#                 result_json = json.loads(result)
#                 return QueryResponse(response=result_json)
#             except json.JSONDecodeError:
#                 return QueryResponse(response={"text": result})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Query processing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Any
# import pandas as pd
# import numpy as np
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global data store
# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def sanitize_python_code(code: str) -> str:
#     """Sanitize and clean Python code."""
#     # Remove backticks and 'python' prefix
#     code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
#     code = re.sub(r"(\s|`)*$", "", code)
#     return code

# def execute_python(code: str) -> str:
#     """Execute Python code safely and return the output."""
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
    
#     # Save current stdout
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         # Make DataFrame available in execution context
#         df = data_store["df"]
#         # Execute the code
#         exec(sanitize_python_code(code))
#         sys.stdout = old_stdout
#         return mystdout.getvalue()
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error executing code: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     """Determine the Vega-Lite data type for a column."""
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     elif pd.api.types.is_datetime64_any_dtype(df[column]):
#         return "temporal"
#     return "nominal"

# def create_visualization(chart_type: str, x: str, y: str = None, aggregate: str = None) -> str:
#     """Create a Vega-Lite visualization specification."""
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {"field": x, "type": get_column_type(df, x)},
#             },
#             "width": "container",
#             "height": 300
#         }
        
#         if y:
#             spec["encoding"]["y"] = {
#                 "field": y,
#                 "type": get_column_type(df, y),
#                 "aggregate": aggregate if aggregate else None
#             }
#         else:
#             spec["encoding"]["y"] = {"aggregate": "count"}
            
#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# def analyze_data(column: str = None, operation: str = None) -> str:
#     """Perform statistical analysis on data."""
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
        
#     try:
#         if column and column not in df.columns:
#             return f"Column '{column}' not found in dataset."
            
#         if operation == "summary":
#             stats = df[column].describe()
#             return f"Summary statistics for {column}:\n{stats.to_string()}"
#         elif operation == "correlation" and column:
#             corr = df.corr()[column]
#             return f"Correlations with {column}:\n{corr.to_string()}"
#         else:
#             return df[column].describe().to_string() if column else df.describe().to_string()
#     except Exception as e:
#         return f"Error analyzing data: {str(e)}"

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that helps analyze data and create visualizations. 
#             Available tools:
#             1. analyze_data(column, operation): Get statistical analysis of data
#             2. create_visualization(chart_type, x, y, aggregate): Create Vega-Lite visualization
#             3. execute_python(code): Execute custom Python code (must use print() for output)
            
#             Think step by step:
#             1. Understand the user's request
#             2. Choose appropriate tool(s)
#             3. Explain your reasoning
#             4. Execute tool(s) and interpret results"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "name": "analyze_data",
#                     "description": "Analyze data statistically",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "column": {"type": "string"},
#                             "operation": {"type": "string", "enum": ["summary", "correlation"]}
#                         }
#                     }
#                 },
#                 {
#                     "name": "create_visualization",
#                     "description": "Create a Vega-Lite visualization",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "point", "boxplot"]},
#                             "x": {"type": "string"},
#                             "y": {"type": "string"},
#                             "aggregate": {"type": "string", "enum": ["mean", "median", "sum", "count"]}
#                         },
#                         "required": ["chart_type", "x"]
#                     }
#                 },
#                 {
#                     "name": "execute_python",
#                     "description": "Execute Python code with pandas DataFrame 'df'",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "code": {"type": "string"}
#                         },
#                         "required": ["code"]
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if function_name == "analyze_data":
#                     result = analyze_data(**arguments)
#                 elif function_name == "create_visualization":
#                     result = create_visualization(**arguments)
#                 elif function_name == "execute_python":
#                     result = execute_python(**arguments)
                
#                 results.append(result)
            
#             # Check if any result is a valid Vega-Lite spec
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             # If no visualization, return text response
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}

# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def sanitize_python_code(code: str) -> str:
#     code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
#     code = re.sub(r"(\s|`)*$", "", code)
#     return code

# def execute_python(code: str) -> str:
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         df = data_store["df"]
#         exec(sanitize_python_code(code))
#         sys.stdout = old_stdout
#         return mystdout.getvalue()
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error executing code: {str(e)}"

# def analyze_data(column: str = None, operation: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
        
#     try:
#         if column and column not in df.columns:
#             return f"Column '{column}' not found in dataset."

#         if operation == "summary":
#             if column:
#                 stats = df[column].describe()
#             else:
#                 stats = df.describe()
#             return f"Summary statistics:\n{stats.to_string()}"
#         elif operation == "correlation" and column:
#             corr = df.corr()[column]
#             return f"Correlations with {column}:\n{corr.to_string()}"
#         else:
#             return df[column].describe().to_string() if column else df.describe().to_string()
#     except Exception as e:
#         return f"Error analyzing data: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     elif pd.api.types.is_datetime64_any_dtype(df[column]):
#         return "temporal"
#     return "nominal"

# def create_visualization(chart_type: str, x: str, y: str = None, aggregate: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {"field": x, "type": get_column_type(df, x)},
#             },
#             "width": "container",
#             "height": 300
#         }
        
#         if y:
#             spec["encoding"]["y"] = {
#                 "field": y,
#                 "type": get_column_type(df, y),
#                 "aggregate": aggregate if aggregate else None
#             }
#         else:
#             spec["encoding"]["y"] = {"aggregate": "count"}
            
#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that helps analyze data and create visualizations. 
#             Available tools:
#             1. analyze_data(column, operation): Get statistical analysis of data
#             2. create_visualization(chart_type, x, y, aggregate): Create Vega-Lite visualization
#             3. execute_python(code): Execute custom Python code (must use print() for output)
            
#             Think step by step and choose the appropriate tool(s) for the user's request."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "analyze_data",
#                         "description": "Analyze data statistically",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "column": {"type": "string"},
#                                 "operation": {"type": "string", "enum": ["summary", "correlation"]}
#                             }
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "point", "boxplot", "histogram"]},
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "aggregate": {"type": "string", "enum": ["mean", "median", "sum", "count"]}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code with pandas DataFrame 'df'",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {"type": "string"}
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if function_name == "analyze_data":
#                     result = analyze_data(**arguments)
#                 elif function_name == "create_visualization":
#                     result = create_visualization(**arguments)
#                 elif function_name == "execute_python":
#                     result = execute_python(**arguments)
                
#                 results.append(result)
            
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}

# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def sanitize_python_code(code: str) -> str:
#     code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
#     code = re.sub(r"(\s|`)*$", "", code)
#     return code

# def execute_python(code: str) -> str:
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
#     try:
#         df = data_store["df"]
#         exec(sanitize_python_code(code))
#         sys.stdout = old_stdout
#         return mystdout.getvalue()
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error executing code: {str(e)}"

# def find_matching_column(df: pd.DataFrame, column: str) -> str:
#     if column in df.columns:
#         return column
    
#     column_lower = column.lower().replace('car_', '').replace('_', ' ')
    
#     # Exact match after normalization
#     for col in df.columns:
#         if col.lower().replace('_', ' ') == column_lower:
#             return col
    
#     # Partial match
#     for col in df.columns:
#         if column_lower in col.lower().replace('_', ' '):
#             return col
    
#     return None

# def analyze_data(column: str = None, operation: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
        
#     try:
#         if column:
#             matching_column = find_matching_column(df, column)
#             if not matching_column:
#                 return f"Column similar to '{column}' not found. Available columns: {', '.join(df.columns)}"
#             column = matching_column

#         if operation == "summary":
#             if column:
#                 stats = df[column].describe()
#             else:
#                 stats = df.describe()
#             return f"Summary statistics:\n{stats.to_string()}"
#         elif operation == "correlation" and column:
#             corr = df.corr()[column]
#             return f"Correlations with {column}:\n{corr.to_string()}"
#         else:
#             if column:
#                 data_range = df[column].max() - df[column].min()
#                 return f"Range of {column}: {df[column].min()} to {df[column].max()} (total range: {data_range})"
#             return df.describe().to_string()
#     except Exception as e:
#         return f"Error analyzing data: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     elif pd.api.types.is_datetime64_any_dtype(df[column]):
#         return "temporal"
#     return "nominal"

# def create_visualization(chart_type: str, x: str, y: str = None, aggregate: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         matching_x = find_matching_column(df, x)
#         if not matching_x:
#             return f"Column similar to '{x}' not found"
        
#         if y:
#             matching_y = find_matching_column(df, y)
#             if not matching_y:
#                 return f"Column similar to '{y}' not found"
#             y = matching_y
        
#         x = matching_x
        
#         spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {"field": x, "type": get_column_type(df, x)},
#             },
#             "width": "container",
#             "height": 300
#         }
        
#         if y:
#             spec["encoding"]["y"] = {
#                 "field": y,
#                 "type": get_column_type(df, y),
#                 "aggregate": aggregate if aggregate else None
#             }
#         else:
#             spec["encoding"]["y"] = {"aggregate": "count"}
            
#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that helps analyze data and create visualizations. 
#             Available tools:
#             1. analyze_data(column, operation): Get statistical analysis of data. The column parameter can be approximate.
#             2. create_visualization(chart_type, x, y, aggregate): Create Vega-Lite visualization
#             3. execute_python(code): Execute custom Python code (must use print() for output)
            
#             For the cars dataset, assume common prefixes like 'car_' can be omitted and focus on the main term.
#             Think step by step and choose the appropriate tool(s) for the user's request."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "analyze_data",
#                         "description": "Analyze data statistically",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "column": {"type": "string"},
#                                 "operation": {"type": "string", "enum": ["summary", "correlation"]}
#                             }
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "point", "boxplot", "histogram"]},
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "aggregate": {"type": "string", "enum": ["mean", "median", "sum", "count"]}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code with pandas DataFrame 'df'",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {"type": "string"}
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if function_name == "analyze_data":
#                     result = analyze_data(**arguments)
#                 elif function_name == "create_visualization":
#                     result = create_visualization(**arguments)
#                 elif function_name == "execute_python":
#                     result = execute_python(**arguments)
                
#                 results.append(result)
            
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def execute_python(code: str) -> str:
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         # Make common libraries available
#         df = data_store["df"]
        
#         cleaned_code = sanitize_python_code(code)
#         exec(cleaned_code)
        
#         sys.stdout = old_stdout
#         output = mystdout.getvalue()
#         return output if output else "Analysis completed, but no output was generated. Make sure to use print() to show results."
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error executing code: {str(e)}"

# def find_matching_column(df: pd.DataFrame, column: str) -> str:
#     if not column:
#         return None
    
#     if column in df.columns:
#         return column
    
#     column_lower = column.lower().replace('car_', '').replace('_', ' ')
    
#     # Exact match after normalization
#     for col in df.columns:
#         if col.lower().replace('_', ' ') == column_lower:
#             return col
            
#     # Partial match
#     for col in df.columns:
#         if column_lower in col.lower().replace('_', ' '):
#             return col
            
#     return None

# def analyze_data(column: str = None, operation: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         if column:
#             matching_column = find_matching_column(df, column)
#             if not matching_column:
#                 return f"Column similar to '{column}' not found. Available columns: {', '.join(df.columns)}"
#             column = matching_column

#         if operation == "summary":
#             if column:
#                 stats = df[column].describe()
#                 stats_dict = stats.to_dict()
#                 return f"Analysis of {column}:\n" + \
#                        f"Average: {stats_dict['mean']:.2f}\n" + \
#                        f"Median: {stats_dict['50%']:.2f}\n" + \
#                        f"Range: {stats_dict['min']:.2f} to {stats_dict['max']:.2f}\n" + \
#                        f"Standard deviation: {stats_dict['std']:.2f}"
#             else:
#                 return df.describe().to_string()
                
#         elif operation == "correlation" and column:
#             corr = df.corr()[column].sort_values(ascending=False)
#             return f"Strongest correlations with {column}:\n" + \
#                    "\n".join([f"{col}: {val:.3f}" for col, val in corr.items() if col != column][:5])
                   
#         else:
#             if column:
#                 series = df[column]
#                 stats = series.describe()
                
#                 # Detect if dealing with whole numbers (like weights)
#                 is_whole_number = all(float(x).is_integer() for x in series if pd.notna(x))
#                 decimals = 0 if is_whole_number else 2
                
#                 # Add units for known columns
#                 unit = " lbs" if "weight" in column.lower() else ""
                
#                 response = f"The range of {column} in the dataset is from "
#                 response += f"{stats['min']:.{decimals}f}{unit} to {stats['max']:.{decimals}f}{unit}."
                
#                 # Add context for numeric columns
#                 if pd.api.types.is_numeric_dtype(series):
#                     response += f"\nThe average is {stats['mean']:.{decimals}f}{unit}, "
#                     response += f"with most values falling between "
#                     response += f"{stats['25%']:.{decimals}f}{unit} and {stats['75%']:.{decimals}f}{unit}."
                
#                 return response
                
#             return df.describe().to_string()
            
#     except Exception as e:
#         return f"Error analyzing data: {str(e)}"

# def create_visualization(chart_type: str, x: str, y: str = None, aggregate: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         matching_x = find_matching_column(df, x)
#         if not matching_x:
#             return f"Column similar to '{x}' not found"
        
#         if y:
#             matching_y = find_matching_column(df, y)
#             if not matching_y:
#                 return f"Column similar to '{y}' not found"
#             y = matching_y
        
#         x = matching_x
        
#         # Handle histogram specifically
#         if chart_type == "histogram":
#             spec = {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": df.to_dict(orient="records")},
#                 "mark": "bar",
#                 "encoding": {
#                     "x": {
#                         "bin": True,
#                         "field": x,
#                         "type": "quantitative"
#                     },
#                     "y": {
#                         "aggregate": "count",
#                         "type": "quantitative"
#                     }
#                 },
#                 "width": "container",
#                 "height": 300
#             }
#         else:
#             spec = {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": df.to_dict(orient="records")},
#                 "mark": chart_type,
#                 "encoding": {
#                     "x": {"field": x, "type": get_column_type(df, x)},
#                 },
#                 "width": "container",
#                 "height": 300
#             }
            
#             if y:
#                 spec["encoding"]["y"] = {
#                     "field": y,
#                     "type": get_column_type(df, y),
#                     "aggregate": aggregate if aggregate else None
#                 }
#             else:
#                 spec["encoding"]["y"] = {"aggregate": "count"}
        
#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     elif pd.api.types.is_datetime64_any_dtype(df[column]):
#         return "temporal"
#     return "nominal"

# def sanitize_python_code(code: str) -> str:
#     code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
#     code = re.sub(r"(\s|`)*$", "", code)
#     return code

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that explains insights naturally and clearly.

# When analyzing data:
# 1. Understand the user's question
# 2. Choose appropriate tools to gather information
# 3. Transform statistical output into natural language
# 4. Provide relevant context and insights
# 5. Combine multiple tool outputs if needed for a complete answer

# For example, instead of raw statistics, say:
# "The average MPG is 20.5, with most cars falling between 15 and 25 MPG."

# Available tools:
# 1. analyze_data(column, operation): Get statistical analysis
# 2. create_visualization(chart_type, x, y, aggregate): Create Vega-Lite visualization
# 3. execute_python(code): Run custom analysis (use print() for output)

# Always explain insights in clear, non-technical language."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "analyze_data",
#                         "description": "Get statistical analysis of data",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "column": {"type": "string"},
#                                 "operation": {
#                                     "type": "string",
#                                     "enum": ["summary", "correlation"]
#                                 }
#                             }
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {
#                                     "type": "string",
#                                     "enum": ["bar", "line", "scatter", "point", "boxplot", "histogram"]
#                                 },
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "aggregate": {
#                                     "type": "string",
#                                     "enum": ["mean", "median", "sum", "count"]
#                                 }
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code with pandas DataFrame 'df'",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {"type": "string"}
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if function_name == "analyze_data":
#                     result = analyze_data(**arguments)
#                 elif function_name == "create_visualization":
#                     result = create_visualization(**arguments)
#                 elif function_name == "execute_python":
#                     result = execute_python(**arguments)
                
#                 results.append(result)
            
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}








# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def format_number(value: float, decimals: int = 2) -> str:
#     if abs(value) >= 1000:
#         return f"{value:,.0f}"
#     return f"{value:.{decimals}f}"

# def process_analytical_query(query: str) -> str:
#     df = data_store["df"]
#     query = query.lower()
    
#     try:
#         if "range" in query and "weight" in query:
#             min_weight = df['Weight'].min()
#             max_weight = df['Weight'].max()
#             return f"The range of car weights in the dataset is from {format_number(min_weight)} to {format_number(max_weight)} lbs."
            
#         elif "median mpg" in query and ("us" in query or "united states" in query):
#             median_mpg = df[df['Origin'].str.lower() == 'us']['MPG'].median()
#             return f"The median MPG for cars from the US is {format_number(median_mpg)}."
            
#         elif "highest horsepower" in query:
#             max_hp = df['Horsepower'].max()
#             car = df[df['Horsepower'] == max_hp]['Model'].iloc[0]
#             return f"The highest horsepower recorded is {format_number(max_hp)}, belonging to the {car}."
            
#         elif "average mpg" in query and "origin" in query:
#             avg_by_origin = df.groupby('Origin')['MPG'].mean()
#             result = "Average MPG by origin:\n" + "\n".join(
#                 f"{origin}: {format_number(mpg)}" for origin, mpg in avg_by_origin.items()
#             )
#             return result
            
#         elif "average mpg" in query and "horsepower" in query and "150" in query:
#             high_hp_mpg = df[df['Horsepower'] > 150]['MPG'].mean()
#             count = len(df[df['Horsepower'] > 150])
#             return f"For cars with horsepower greater than 150 ({count} cars), the average MPG is {format_number(high_hp_mpg)}."
            
#         return None
#     except Exception as e:
#         logger.error(f"Error in analytical query: {str(e)}")
#         return None

# def execute_python(code: str) -> str:
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         df = data_store["df"]
#         exec(sanitize_python_code(code))
#         sys.stdout = old_stdout
#         output = mystdout.getvalue()
#         return output if output else "Analysis completed successfully."
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error in analysis: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     elif pd.api.types.is_datetime64_any_dtype(df[column]):
#         return "temporal"
#     return "nominal"

# def create_visualization(chart_type: str, x: str, y: str = None, aggregate: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {"field": x, "type": get_column_type(df, x)},
#             },
#             "width": "container",
#             "height": 300
#         }
        
#         if y:
#             spec["encoding"]["y"] = {
#                 "field": y,
#                 "type": get_column_type(df, y),
#                 "aggregate": aggregate if aggregate else None
#             }
#         else:
#             spec["encoding"]["y"] = {"aggregate": "count"}
            
#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# def sanitize_python_code(code: str) -> str:
#     code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
#     code = re.sub(r"(\s|`)*$", "", code)
#     return code

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     # Try direct analytical queries first
#     direct_response = process_analytical_query(request.prompt)
#     if direct_response:
#         return QueryResponse(response={"text": direct_response})
    
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant. For data questions:
#             1. Use execute_python() for calculations
#             2. Use create_visualization() for charts
#             3. Explain results clearly
#             4. Use print() to show results in execute_python()"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Run Python code using pandas DataFrame 'df'",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {"type": "string"}
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a data visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "boxplot", "histogram"]},
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "aggregate": {"type": "string", "enum": ["mean", "median", "sum", "count"]}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if function_name == "execute_python":
#                     result = execute_python(**arguments)
#                 elif function_name == "create_visualization":
#                     result = create_visualization(**arguments)
                
#                 results.append(result)
            
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def format_number(value: float, decimals: int = 2) -> str:
#     if abs(value) >= 1000:
#         return f"{value:,.0f}"
#     return f"{value:.{decimals}f}"

# def create_visualization(chart_type: str, x: str, y: str = None, 
#                        aggregate: str = None, color: str = None, 
#                        facet: str = None) -> str:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {"field": x, "type": get_column_type(df, x)},
#             },
#             "width": "container",
#             "height": 300
#         }

#         # Handle y-axis
#         if y:
#             y_spec = {
#                 "field": y,
#                 "type": get_column_type(df, y)
#             }
#             if aggregate:
#                 y_spec["aggregate"] = aggregate
#             spec["encoding"]["y"] = y_spec
#         else:
#             spec["encoding"]["y"] = {"aggregate": "count"}

#         # Add color encoding if specified
#         if color:
#             spec["encoding"]["color"] = {
#                 "field": color,
#                 "type": get_column_type(df, color)
#             }

#         # Add faceting if specified
#         if facet:
#             spec["facet"] = {
#                 "field": facet,
#                 "type": get_column_type(df, facet)
#             }
#             spec["columns"] = 2

#         # Special handling for histograms
#         if chart_type == "histogram":
#             spec["mark"] = "bar"
#             spec["encoding"]["x"]["bin"] = True
#             spec["encoding"]["y"] = {"aggregate": "count"}

#         return json.dumps(spec)
#     except Exception as e:
#         return f"Error creating visualization: {str(e)}"

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     return "nominal"

# def process_visualization_query(query: str) -> str:
#     query = query.lower()
#     df = data_store["df"]
    
#     try:
#         if "mpg across origins" in query or "mpg by origin" in query:
#             return create_visualization("bar", "Origin", "MPG", "mean")
            
#         if "weight and acceleration" in query:
#             return create_visualization("scatter", "Weight", "Acceleration")
            
#         if "mpg and horsepower" in query:
#             return create_visualization("scatter", "Horsepower", "MPG")
            
#         if "number of cars from each" in query and ("origin" in query or "country" in query):
#             return create_visualization("bar", "Origin")
            
#         if "acceleration" in query and "cylinders" in query and "origin" in query:
#             return create_visualization("bar", "Cylinders", "Acceleration", "mean", "Origin", "Cylinders")
            
#         if "displacement" in query and "mpg" in query:
#             return create_visualization("scatter", "Displacement", "MPG", color="Origin")
            
#         if "cylinders" in query and "mpg" in query:
#             return create_visualization("bar", "Cylinders", "MPG", "mean")

#         return None
#     except Exception as e:
#         logger.error(f"Error in visualization query: {str(e)}")
#         return None

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     # Try visualization queries first
#     viz_response = process_visualization_query(request.prompt)
#     if viz_response:
#         try:
#             spec = json.loads(viz_response)
#             return QueryResponse(response=spec)
#         except:
#             pass

#     try:
#         messages = [
#             {"role": "system", "content": """You are a data visualization assistant. For visualization requests:
#             1. Use create_visualization() with appropriate parameters
#             2. Support bar charts, scatter plots, and histograms
#             3. Handle aggregations and grouping
#             4. Use color encoding for additional dimensions"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a data visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {
#                                     "type": "string",
#                                     "enum": ["bar", "scatter", "histogram", "line", "boxplot"]
#                                 },
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "aggregate": {
#                                     "type": "string",
#                                     "enum": ["mean", "median", "sum", "count"]
#                                 },
#                                 "color": {"type": "string"},
#                                 "facet": {"type": "string"}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")

#         if response.tool_calls:
#             results = []
#             for tool_call in response.tool_calls:
#                 arguments = json.loads(tool_call.function.arguments)
#                 result = create_visualization(**arguments)
#                 results.append(result)
            
#             for result in results:
#                 try:
#                     spec = json.loads(result)
#                     if isinstance(spec, dict) and spec.get("mark"):
#                         return QueryResponse(response=spec)
#                 except json.JSONDecodeError:
#                     continue
            
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def format_number(value: float) -> str:
#     """Format numbers with commas and appropriate decimals."""
#     if pd.isna(value):
#         return "N/A"
#     if abs(value) >= 1000:
#         return f"{value:,.0f}"
#     return f"{value:.2f}"

# def process_simple_query(query: str) -> str:
#     """Handle straightforward analytical queries with concise responses."""
#     df = data_store["df"]
#     query = query.lower()
    
#     try:
#         if "range" in query and "weight" in query:
#             min_weight = df['Weight'].min()
#             max_weight = df['Weight'].max()
#             return f"The range of car weights in the dataset is from {format_number(min_weight)} to {format_number(max_weight)} lbs."
            
#         elif "median mpg" in query and ("us" in query or "united states" in query):
#             median_mpg = df[df['Origin'].str.lower() == 'us']['MPG'].median()
#             return f"The median MPG for cars from the US is {format_number(median_mpg)}."
            
#         elif "highest horsepower" in query:
#             max_hp = df['Horsepower'].max()
#             car = df[df['Horsepower'] == max_hp]['Model'].iloc[0]
#             return f"The highest horsepower recorded is {format_number(max_hp)}, belonging to the {car.lower()}."
            
#         return None
#     except Exception as e:
#         logger.error(f"Error in simple query: {str(e)}")
#         return None

# def create_visualization(chart_type: str, x: str, y: str = None, color: str = None, 
#                        aggregate: str = None, facet: str = None) -> dict:
#     """Create Vega-Lite visualization specification."""
#     df = data_store["df"]
    
#     spec = {
#         "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#         "data": {"values": df.to_dict(orient="records")},
#         "mark": chart_type,
#         "encoding": {
#             "x": {"field": x, "type": get_column_type(df, x)},
#         },
#         "width": "container",
#         "height": 300
#     }
    
#     if y:
#         y_spec = {
#             "field": y,
#             "type": get_column_type(df, y)
#         }
#         if aggregate:
#             y_spec["aggregate"] = aggregate
#         spec["encoding"]["y"] = y_spec
#     else:
#         spec["encoding"]["y"] = {"aggregate": "count"}

#     if color:
#         spec["encoding"]["color"] = {
#             "field": color,
#             "type": get_column_type(df, color)
#         }

#     if facet:
#         spec["facet"] = {
#             "field": facet,
#             "type": get_column_type(df, facet)
#         }

#     if chart_type == "histogram":
#         spec["mark"] = "bar"
#         spec["encoding"]["x"]["bin"] = True
#         spec["encoding"]["y"] = {"aggregate": "count"}

#     return spec

# def process_visualization_query(query: str) -> dict:
#     """Handle visualization requests with appropriate chart types."""
#     query = query.lower()
    
#     try:
#         if any(x in query for x in ["average mpg", "mpg across origins"]):
#             return create_visualization("bar", "Origin", "MPG", aggregate="mean")
            
#         if "weight" in query and "acceleration" in query:
#             return create_visualization("scatter", "Weight", "Acceleration")
            
#         if "mpg" in query and "horsepower" in query:
#             return create_visualization("scatter", "Horsepower", "MPG")
            
#         if "breakdown" in query and "origin" in query:
#             return create_visualization("bar", "Origin")
            
#         if "histogram" in query and "mpg" in query:
#             return create_visualization("histogram", "MPG")

#         return None
#     except Exception as e:
#         logger.error(f"Error in visualization query: {str(e)}")
#         return None

# def get_column_type(df: pd.DataFrame, column: str) -> str:
#     """Determine the data type for Vega-Lite specification."""
#     if pd.api.types.is_numeric_dtype(df[column]):
#         return "quantitative"
#     return "nominal"

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     # Try simple queries first
#     simple_response = process_simple_query(request.prompt)
#     if simple_response:
#         return QueryResponse(response={"text": simple_response})
    
#     # Try visualization queries next
#     viz_spec = process_visualization_query(request.prompt)
#     if viz_spec:
#         return QueryResponse(response=viz_spec)
    
#     # Fall back to GPT-4 for complex queries
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that provides concise responses and visualizations. 
#             For simple queries, give direct answers. For complex analyses, create appropriate visualizations."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a data visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {"type": "string", "enum": ["bar", "scatter", "histogram", "line", "boxplot"]},
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "color": {"type": "string"},
#                                 "aggregate": {"type": "string", "enum": ["mean", "median", "sum", "count"]}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
        
#         if response.tool_calls:
#             arguments = json.loads(response.tool_calls[0].function.arguments)
#             viz_spec = create_visualization(**arguments)
#             return QueryResponse(response=viz_spec)

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         return {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}

# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any, Optional
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def format_number(value: float) -> str:
#     if pd.isna(value):
#         return "N/A"
#     if abs(value) >= 1000:
#         return f"{value:,.0f}"
#     return f"{value:.2f}"

# def create_visualization(chart_type: str, x: str, y: str = None, color: str = None) -> dict:
#     df = data_store["df"]
#     if df is None:
#         return "No data loaded."
    
#     try:
#         base_spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "width": "container",
#             "height": 300,
#             "data": {"values": df.to_dict(orient="records")},
#             "mark": chart_type,
#             "encoding": {
#                 "x": {
#                     "field": x,
#                     "type": "quantitative",
#                     "title": x
#                 }
#             }
#         }

#         if y:
#             base_spec["encoding"]["y"] = {
#                 "field": y,
#                 "type": "quantitative",
#                 "title": y
#             }

#         if color:
#             base_spec["encoding"]["color"] = {
#                 "field": color,
#                 "type": "nominal"
#             }

#         if chart_type == "point":
#             corr = df[x].corr(df[y])
#             correlation_text = f"Correlation: {corr:.2f}"
            
#             base_spec["layer"] = [
#                 {"mark": "point"},
#                 {
#                     "mark": "line",
#                     "transform": [{"regression": y, "on": x}],
#                     "encoding": {
#                         "color": {"value": "red"}
#                     }
#                 }
#             ]
#             base_spec["title"] = correlation_text

#         return base_spec

#     except Exception as e:
#         logger.error(f"Error creating visualization: {str(e)}")
#         return str(e)

# def process_analytical_query(query: str) -> str:
#     df = data_store["df"]
#     query = query.lower()
    
#     try:
#         if "range" in query and "weight" in query:
#             min_weight = df['Weight'].min()
#             max_weight = df['Weight'].max()
#             return f"The range of car weights in the dataset is from {format_number(min_weight)} to {format_number(max_weight)} lbs."
            
#         elif "median mpg" in query and ("us" in query or "united states" in query):
#             median_mpg = df[df['Origin'].str.lower() == 'us']['MPG'].median()
#             return f"The median MPG for cars from the US is {format_number(median_mpg)}."
            
#         elif "highest horsepower" in query:
#             max_hp = df['Horsepower'].max()
#             car = df[df['Horsepower'] == max_hp]['Model'].iloc[0]
#             return f"The highest horsepower recorded is {format_number(max_hp)}, belonging to the {car.lower()}."
            
#         return None
#     except Exception as e:
#         logger.error(f"Error in analytical query: {str(e)}")
#         return None

# def process_visualization_query(query: str) -> Optional[Dict]:
#     query = query.lower()
    
#     try:
#         if "mpg" in query and "horsepower" in query and "scatter" in query:
#             return create_visualization("point", "Horsepower", "MPG")
            
#         if "weight" in query and "acceleration" in query and "scatter" in query:
#             return create_visualization("point", "Weight", "Acceleration")
            
#         if "mpg" in query and "histogram" in query:
#             return create_visualization("bar", "MPG", None)
            
#         if "origin" in query and "bar" in query:
#             return create_visualization("bar", "Origin", None)
            
#         if "average mpg" in query and "origin" in query:
#             return create_visualization("bar", "Origin", "MPG", aggregate="mean")
            
#     except Exception as e:
#         logger.error(f"Error in visualization query: {str(e)}")
#         return None

# def calculate_correlation_summary(x: str, y: str) -> str:
#     df = data_store["df"]
#     corr = df[x].corr(df[y])
#     slope, intercept, r_value, p_value, std_err = stats.linregress(df[x], df[y])
    
#     strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
#     direction = "positive" if corr > 0 else "negative"
    
#     return f"There is a {strength} {direction} correlation ({corr:.2f}) between {x} and {y}."

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     analytical_response = process_analytical_query(request.prompt)
#     if analytical_response:
#         return QueryResponse(response={"text": analytical_response})
    
#     viz_spec = process_visualization_query(request.prompt)
#     if viz_spec:
#         if "scatter" in request.prompt.lower() or "correlation" in request.prompt.lower():
#             x = viz_spec["encoding"]["x"]["field"]
#             y = viz_spec["encoding"]["y"]["field"]
#             correlation_summary = calculate_correlation_summary(x, y)
#             viz_spec["title"] = correlation_summary
#         return QueryResponse(response=viz_spec)
    
#     try:
#         messages = [
#             {"role": "system", "content": """You are a data analysis assistant that provides visualizations and insights.
#             For scatterplots, include correlation analysis. For distributions, include summary statistics."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a data visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "chart_type": {
#                                     "type": "string",
#                                     "enum": ["point", "bar", "line", "boxplot", "histogram"]
#                                 },
#                                 "x": {"type": "string"},
#                                 "y": {"type": "string"},
#                                 "color": {"type": "string"}
#                             },
#                             "required": ["chart_type", "x"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
        
#         if response.tool_calls:
#             arguments = json.loads(response.tool_calls[0].function.arguments)
#             viz_spec = create_visualization(**arguments)
#             return QueryResponse(response=viz_spec)

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         return {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any, Optional
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def execute_python(code: str) -> str:
#     """Execute Python code and capture its output."""
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a CSV file first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         # Make DataFrame and libraries available
#         df = data_store["df"]
#         exec(code)
#         sys.stdout = old_stdout
#         output = mystdout.getvalue()
#         return output if output else "Analysis completed successfully."
#     except Exception as e:
#         sys.stdout = old_stdout
#         return f"Error in analysis: {str(e)}"

# def create_visualization(spec: Dict) -> Dict:
#     """Create a Vega-Lite visualization with the given specification."""
#     df = data_store["df"]
    
#     try:
#         # Add the data to the spec
#         spec["data"] = {"values": df.to_dict(orient="records")}
        
#         # Set default width and height if not specified
#         spec.setdefault("width", "container")
#         spec.setdefault("height", 300)
        
#         # Add schema
#         spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        
#         return spec
#     except Exception as e:
#         logger.error(f"Error creating visualization: {str(e)}")
#         return {"error": str(e)}

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     try:
#         # Get dataset info for context
#         df = data_store["df"]
#         if df is None:
#             return QueryResponse(response={"text": "Please upload a dataset first."})
        
#         column_info = {col: str(df[col].dtype) for col in df.columns}
        
#         messages = [
#             {"role": "system", "content": f"""You are a data analysis assistant. You have access to a dataset with the following columns:
#             {json.dumps(column_info, indent=2)}
            
#             You can:
#             1. Execute Python code using pandas (df is the DataFrame)
#             2. Create visualizations using Vega-Lite specs
            
#             For statistical queries:
#             - Use Python code to calculate exact values
#             - Format numbers nicely
#             - Provide concise, natural language responses
            
#             For visualization requests:
#             - Create appropriate Vega-Lite specs
#             - Include statistical summaries when relevant
#             - Use color encoding for additional dimensions
            
#             Always print() results in Python code."""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code using pandas (df is the DataFrame)",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {
#                                     "type": "string",
#                                     "description": "Python code to execute (use print() for output)"
#                                 }
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "spec": {
#                                     "type": "object",
#                                     "description": "Vega-Lite specification"
#                                 }
#                             },
#                             "required": ["spec"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
        
#         if response.tool_calls:
#             results = []
#             viz_spec = None
            
#             for tool_call in response.tool_calls:
#                 arguments = json.loads(tool_call.function.arguments)
                
#                 if tool_call.function.name == "execute_python":
#                     result = execute_python(arguments["code"])
#                     if result:
#                         results.append(result)
                        
#                 elif tool_call.function.name == "create_visualization":
#                     viz_spec = create_visualization(arguments["spec"])
            
#             # Return visualization if one was created
#             if viz_spec and not viz_spec.get("error"):
#                 if results:
#                     viz_spec["title"] = "\n".join(results)
#                 return QueryResponse(response=viz_spec)
            
#             # Otherwise return text results
#             return QueryResponse(response={"text": "\n".join(results)})

#         return QueryResponse(response={"text": response.content})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         return {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}

# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any, Optional
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global data store
# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def execute_python(code: str) -> str:
#     """Execute Python code and capture its output."""
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a dataset first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         # Make DataFrame and libraries available
#         df = data_store["df"]
#         exec(code)
#         sys.stdout = old_stdout
#         output = mystdout.getvalue()
#         return output if output else "Analysis completed successfully."
#     except Exception as e:
#         sys.stdout = old_stdout
#         logger.error(f"Python execution error: {str(e)}")
#         return f"Error in analysis: {str(e)}"

# def format_number(value: float) -> str:
#     """Format numbers with appropriate precision and commas."""
#     if pd.isna(value):
#         return "N/A"
#     if abs(value) >= 1000:
#         return f"{value:,.0f}"
#     return f"{value:.2f}"

# def create_visualization(spec: Dict) -> Dict:
#     """Create a Vega-Lite visualization with the given specification."""
#     df = data_store["df"]
    
#     try:
#         base_spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "width": "container",
#             "height": 300,
#             "data": {"values": df.to_dict(orient="records")}
#         }
        
#         # Merge the provided spec with base spec
#         final_spec = {**base_spec, **spec}
        
#         # Ensure proper encoding types
#         if "encoding" in final_spec:
#             for axis in ["x", "y"]:
#                 if axis in final_spec["encoding"]:
#                     field = final_spec["encoding"][axis].get("field")
#                     if field and field in df.columns:
#                         if pd.api.types.is_numeric_dtype(df[field]):
#                             final_spec["encoding"][axis]["type"] = "quantitative"
#                         else:
#                             final_spec["encoding"][axis]["type"] = "nominal"

#         return final_spec
#     except Exception as e:
#         logger.error(f"Visualization error: {str(e)}")
#         return {"error": str(e)}

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     try:
#         df = data_store["df"]
#         if df is None:
#             return QueryResponse(response={"text": "Please upload a dataset first."})
        
#         # Get dataset information
#         column_info = {col: str(df[col].dtype) for col in df.columns}
#         sample_data = df.head(3).to_dict(orient="records")
        
#         # Update the system prompt in the /query endpoint
#         messages = [
#             {"role": "system", "content": f"""You are a data analysis assistant. You have access to a dataset with these columns:
#             {json.dumps(column_info, indent=2)}
            
#             Sample data:
#             {json.dumps(sample_data, indent=2)}
            
#             For statistical analysis, write Python code like this:
#             ```python
#             # Calculate range
#             column_name = 'Weight'  # or any other column
#             min_value = df[column_name].min()
#             max_value = df[column_name].max()
#             print(f"The range of {column_name} is from {min_value:,.0f} to {max_value:,.0f}")
            
#             # Calculate averages by group
#             group_col = 'Origin'  # or any grouping column
#             value_col = 'MPG'    # or any value column
#             result = df.groupby(group_col)[value_col].mean()
#             for group, value in result.items():
#                 print(f"{group}: {value:.2f}")
#             ```
            
#             For visualizations, return a Vega-Lite spec like this:
#             ```json
#             {
#                 "mark": "bar",
#                 "encoding": {
#                     "x": {"field": "Origin", "type": "nominal"},
#                     "y": {"field": "MPG", "type": "quantitative", "aggregate": "mean"},
#                     "color": {"field": "Origin", "type": "nominal"}
#                 }
#             }
#             ```
            
#             Always:
#             1. Use execute_python() for calculations
#             2. Use create_visualization() for charts
#             3. Use proper Python variable names (not placeholders)
#             4. Print results clearly with context
#             5. Include units where appropriate (e.g., lbs for weight)"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code using pandas (df is the DataFrame)",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {
#                                     "type": "string",
#                                     "description": "Python code to execute (use print() for output)"
#                                 }
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "spec": {
#                                     "type": "object",
#                                     "properties": {
#                                         "mark": {"type": "string"},
#                                         "encoding": {"type": "object"},
#                                         "title": {"type": "string"}
#                                     },
#                                     "required": ["mark", "encoding"]
#                                 }
#                             },
#                             "required": ["spec"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")
        
#         if response.tool_calls:
#             results = []
#             viz_spec = None
            
#             for tool_call in response.tool_calls:
#                 try:
#                     arguments = json.loads(tool_call.function.arguments)
                    
#                     if tool_call.function.name == "execute_python":
#                         result = execute_python(arguments["code"])
#                         if result:
#                             results.append(result)
                            
#                     elif tool_call.function.name == "create_visualization":
#                         viz_spec = create_visualization(arguments["spec"])
#                 except json.JSONDecodeError as e:
#                     logger.error(f"JSON decode error: {e}")
#                     continue
#                 except Exception as e:
#                     logger.error(f"Tool execution error: {e}")
#                     continue
            
#             # Return visualization if one was created
#             if viz_spec and not viz_spec.get("error"):
#                 if results:
#                     viz_spec["title"] = "\n".join(results)
#                 return QueryResponse(response=viz_spec)
            
#             # Return text results or error message
#             if results:
#                 return QueryResponse(response={"text": "\n".join(results)})
#             return QueryResponse(response={"text": "Sorry, I couldn't process that request. Please try rephrasing it."})

#         if response.content:
#             return QueryResponse(response={"text": response.content})
            
#         return QueryResponse(response={"text": "I couldn't understand how to analyze that. Please try rephrasing your question."})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return QueryResponse(response={"text": "An error occurred while processing your request. Please try again."})

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any, Optional
# import pandas as pd
# import numpy as np
# from scipy import stats
# import io
# import json
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import logging
# import re
# from io import StringIO
# import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global data store
# data_store = {"df": None}

# class QueryRequest(BaseModel):
#     prompt: str

# class QueryResponse(BaseModel):
#     response: Dict[str, Any]

# def execute_python(code: str) -> str:
#     """Execute Python code and capture its output."""
#     if data_store["df"] is None:
#         return "No data loaded. Please upload a dataset first."
    
#     old_stdout = sys.stdout
#     sys.stdout = mystdout = StringIO()
    
#     try:
#         df = data_store["df"]
#         exec(code)
#         sys.stdout = old_stdout
#         output = mystdout.getvalue()
#         return output if output else "Analysis completed successfully."
#     except Exception as e:
#         sys.stdout = old_stdout
#         logger.error(f"Python execution error: {str(e)}")
#         return f"Error in analysis: {str(e)}"

# def create_visualization(spec: Dict) -> Dict:
#     """Create a Vega-Lite visualization with the given specification."""
#     df = data_store["df"]
    
#     try:
#         base_spec = {
#             "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#             "width": "container",
#             "height": 300,
#             "data": {"values": df.to_dict(orient="records")}
#         }
        
#         final_spec = {**base_spec, **spec}
        
#         if "encoding" in final_spec:
#             for axis in ["x", "y"]:
#                 if axis in final_spec["encoding"]:
#                     field = final_spec["encoding"][axis].get("field")
#                     if field and field in df.columns:
#                         if pd.api.types.is_numeric_dtype(df[field]):
#                             final_spec["encoding"][axis]["type"] = "quantitative"
#                         else:
#                             final_spec["encoding"][axis]["type"] = "nominal"

#         return final_spec
#     except Exception as e:
#         logger.error(f"Visualization error: {str(e)}")
#         return {"error": str(e)}

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     logger.info(f"Received query: {request.prompt}")
    
#     try:
#         df = data_store["df"]
#         if df is None:
#             return QueryResponse(response={"text": "Please upload a dataset first."})
        
#         column_info = {col: str(df[col].dtype) for col in df.columns}
#         sample_data = df.head(3).to_dict(orient="records")
        
#         messages = [
#             {"role": "system", "content": f"""You are a data analysis assistant. You have access to a dataset with these columns:
# {json.dumps(column_info, indent=2)}

# Sample data:
# {json.dumps(sample_data, indent=2)}

# For statistical queries, write complete Python code like these examples:

# Example 1 - Finding ranges:
# ```python
# min_weight = df['Weight'].min()
# max_weight = df['Weight'].max()
# print(f"The range of car weights is from {{min_weight:,.0f}} to {{max_weight:,.0f}} lbs")
# ```

# Example 2 - Calculating averages by group:
# ```python
# avg_mpg = df.groupby('Origin')['MPG'].mean()
# for origin, mpg in avg_mpg.items():
#     print(f"{{origin}}: {{mpg:.1f}} MPG")
# ```

# Example 3 - Finding medians with conditions:
# ```python
# us_mpg = df[df['Origin'] == 'US']['MPG'].median()
# print(f"The median MPG for US cars is {{us_mpg:.1f}}")
# ```

# For visualizations, return a Vega-Lite spec like these examples:

# Example 1 - Bar chart:
# ```json
# {
#     "mark": "bar",
#     "encoding": {
#         "x": {"field": "Origin", "type": "nominal"},
#         "y": {"field": "MPG", "type": "quantitative", "aggregate": "mean"}
#     }
# }
# ```

# Example 2 - Scatter plot:
# ```json
# {
#     "mark": "point",
#     "encoding": {
#         "x": {"field": "Weight", "type": "quantitative"},
#         "y": {"field": "MPG", "type": "quantitative"},
#         "color": {"field": "Origin", "type": "nominal"}
#     }
# }
# ```

# Example 3 - Histogram:
# ```json
# {
#     "mark": "bar",
#     "encoding": {
#         "x": {"field": "MPG", "bin": true},
#         "y": {"aggregate": "count"}
#     }
# }
# ```

# Important:
# 1. Write complete, executable code (no placeholders)
# 2. Use actual column names from the dataset
# 3. Include units in output (MPG, lbs, etc.)
# 4. Format numbers with commas for thousands
# 5. For correlations, include statistical context
# 6. For visualizations, choose appropriate chart types"""},
#             {"role": "user", "content": request.prompt}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "execute_python",
#                         "description": "Execute Python code using pandas (df is the DataFrame)",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "code": {
#                                     "type": "string",
#                                     "description": "Python code to execute (use print() for output)"
#                                 }
#                             },
#                             "required": ["code"]
#                         }
#                     }
#                 },
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "create_visualization",
#                         "description": "Create a Vega-Lite visualization",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "spec": {
#                                     "type": "object",
#                                     "properties": {
#                                         "mark": {"type": "string"},
#                                         "encoding": {"type": "object"},
#                                         "title": {"type": "string"}
#                                     },
#                                     "required": ["mark", "encoding"]
#                                 }
#                             },
#                             "required": ["spec"]
#                         }
#                     }
#                 }
#             ],
#             tool_choice="auto"
#         )

#         response = completion.choices[0].message
#         logger.info(f"GPT response: {response}")
        
#         if response.tool_calls:
#             results = []
#             viz_spec = None
            
#             for tool_call in response.tool_calls:
#                 try:
#                     arguments = json.loads(tool_call.function.arguments)
                    
#                     if tool_call.function.name == "execute_python":
#                         result = execute_python(arguments["code"])
#                         if result:
#                             results.append(result)
                            
#                     elif tool_call.function.name == "create_visualization":
#                         viz_spec = create_visualization(arguments["spec"])
#                 except json.JSONDecodeError as e:
#                     logger.error(f"JSON decode error: {e}")
#                     continue
#                 except Exception as e:
#                     logger.error(f"Tool execution error: {e}")
#                     continue
            
#             if viz_spec and not viz_spec.get("error"):
#                 if results:
#                     viz_spec["title"] = "\n".join(results)
#                 return QueryResponse(response=viz_spec)
            
#             if results:
#                 return QueryResponse(response={"text": "\n".join(results)})
#             return QueryResponse(response={"text": "Sorry, I couldn't process that request. Please try rephrasing it."})

#         if response.content:
#             return QueryResponse(response={"text": response.content})
            
#         return QueryResponse(response={"text": "I couldn't understand how to analyze that. Please try rephrasing your question."})

#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return QueryResponse(response={"text": "An error occurred while processing your request. Please try again."})

# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     logger.info(f"Receiving file: {file.filename}")
    
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#         data_store["df"] = df
        
#         response = {
#             "message": "File uploaded successfully",
#             "columns": df.columns.tolist(),
#             "sample": df.head().to_dict(orient="records")
#         }
#         logger.info(f"File processed successfully: {df.shape}")
#         return response
#     except Exception as e:
#         logger.error(f"File upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"status": "running"}


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
    allow_origins=["http://localhost:3000"],
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