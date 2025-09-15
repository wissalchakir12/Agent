import sys
print("PYTHONPATH:", sys.path)
import os
from exa_py import Exa
from textwrap import dedent
from typing import Iterator
from dotenv import load_dotenv
from agno.workflow import Workflow
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools
from agno.utils.pprint import pprint_run_response

load_dotenv()

### Exa Setup ###
exa_api_key = os.environ.get('EXA_API_KEY')

exa = Exa(api_key = exa_api_key)

# Exa search function tool
def exa_search(product_list: str, location: str):
    taskStub = exa.research.create_task(
    instructions = f"""
    You are an expert in business procurement research. Your task is to help a company find the best vendors and prices for its needed products.

    Instructions:

    1. Search for each product in the provided list.
    2. Prioritize vendors that operate in or deliver to the specified country and city.
    3. For each product, return details for the top 3 relevant vendors:
        - Vendor Name
        - Product Title
        - Price (in local currency if possible)
        - Vendor Website or Purchase Link
        - Short Description of the product (highlight key features/use)
    4. If available, provide:
        - Minimum order quantity
        - Shipping time
        - Known bulk discounts or deals
    5. Ensure vendors are reliable and legit. Prioritize verified marketplaces, distributors, or direct manufacturers.

    Input:
    - Product List: {product_list}
    - Location: {location}

    Output Format:
    Use clean, structured markdown with clear sections for each product and vendor. Organize the information for easy comparison.
    """,
    model = "exa-research",
    output_infer_schema = False
    )

    # Research can also be used (without an output schema)
    # directly inside chat completions
    from openai import OpenAI

    client = OpenAI(
        base_url = "https://api.exa.ai",
        api_key = exa_api_key,
    )

    completion = client.chat.completions.create(
        model = "exa-research",
        messages = [
            {"role": "user", "content": f"""
            You are an expert in business procurement research. Your task is to find the best vendors and prices for the following products:

            - Product List: {product_list}
            - Location: {location}

            Return the results in clean, structured markdown format. For each product, list the top 3 vendors with:
            - Vendor Name
            - Product Title
            - Price (in local currency)
            - Vendor Website or Purchase Link
            - Short Description
            - Minimum Order Quantity (if available)
            - Shipping Time (if available)
            - Known Bulk Discounts or Deals (if available)

            Organize the output clearly for comparison.
            """}
        ],
        stream = True,
    )

    full_content = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
    print("-------------------------------- EXA SEARCH COMPLETION --------------------------------")
    print(full_content)
    print("------------------------------------------------------------------------------------------------")
    return full_content

class ProcurementAgent(Workflow):
    procurement_agent: Agent = Agent(
        name='Procurement Agent',
        model=OpenAIChat(id='gpt-4o'),
        instructions=dedent("""
        You are a procurement analysis agent.

        Your goal is to:
        1. Parse the markdown-formatted research output from Exa.
        2. For each product listed, extract the following fields for each vendor:
            - Product Name
            - Vendor Name
            - Product Title
            - Price (convert to numeric if possible)
            - Currency
            - Vendor Website or Purchase Link
            - Short Product Description
            - Minimum Order Quantity (if available)
            - Shipping Time (if available)
            - Bulk Discounts or Deals (if available)
            - Vendor Location (if mentioned)

        3. After extracting the data, write and execute a Python script that creates a file named `data.csv`
            - Use the standard `csv` module
            - After each product, leave a blank line
            - Make the columns in this order: Product Name, Vendor Name, Product Title, Price, Currency, Bulk Discounts or Deals, Vendor Website, Short Product Description, Minimum Order Quantity, Shipping Time
            - The script should write a header row followed by one row per vendor
            - IMPORTANT: When opening the file, always use encoding='utf-8' in open(), e.g. open('data.csv', 'w', newline='', encoding='utf-8')
            - Then use the PythonTools tool to run the script and save the data
            - Based on the data given, create rows for each data point

        Then:

        4. Analyze the data across all products and vendors.
            - Compare vendors based on pricing, shipping times, minimum quantities, and available deals.
            - Prioritize vendors who:
                - Deliver to the specified location
                - Offer the lowest price for comparable quality
                - Have favorable shipping times or bulk deals
                - Appear reliable (from marketplaces or verified sellers)

        5. Write an executive summary that includes:
            - Recommended vendor(s) per product
            - Reasons for the recommendation (price, location, delivery, etc.)
            - Any noteworthy observations (e.g. big pricing differences, best bundle offers)
            - Optional: Flag any vendors that should be avoided due to missing info or suspicious listings

        IMPORTANT: Use PythonTools to write the extracted data to `data.csv`.
        """),
        expected_output=dedent("""
        The output should include:

        1. 📊 Data Summary:
        - Number of products processed
        - Total vendors compared
        - Location considered for delivery: <city>, <country>
        - Any data quality issues or missing fields (if relevant)

        2. 🏆 Recommendations Per Product:
        For each product (e.g., "Office Chair", "Laptop"), provide:

        ### Product: <Product Name>

        **Recommended Vendor:** <Vendor Business Name>  
        **Price:** <Price and Currency>  
        **Why Chosen:**  
        - Reason 1 (e.g. best price for similar features)
        - Reason 2 (e.g. fastest shipping)
        - Reason 3 (e.g. known/verified vendor or bulk deal)

        **Runner-Up:** <Vendor Business Name>  
        - Mention if relevant (e.g. slightly higher price but faster delivery or better reviews)

        IMPORTANT: Confirm that the CSV file was written as part of this run.
        """),
        tools=[PythonTools()],
        show_tool_calls=True,
        markdown=True
    )

    def run(self, product_list: str, location: str):
        research_response = exa_search(product_list, location)
        # test_response = """
        # Product: Office Chair
        # Vendor: Office Chair Vendor
        # Price: $100
        # Currency: USD
        # Vendor Website: https://www.officechairvendor.com
        # Short Product Description: A comfortable office chair with a modern design.
        # """
        yield from self.procurement_agent.run(research_response, stream=True)

if __name__ == '__main__':
    from rich.prompt import Prompt
    product_list = Prompt.ask('Enter your products list seperated by commas')
    location = Prompt.ask('Enter your business location (city, country)')
    if product_list and location:
        workflow = ProcurementAgent()
        response: Iterator[RunResponse] = workflow.run(product_list=product_list, location=location)
        pprint_run_response(response, markdown=True)

# --- FastAPI Web Server ---
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import tempfile
import os
import re
import ast
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/procure")
async def procure(product_list: str = Form(...), location: str = Form(...)):
    workflow = ProcurementAgent()
    response: Iterator[RunResponse] = workflow.run(product_list=product_list, location=location)
    # Collect the streamed markdown output
    markdown_output = ""
    for chunk in response:
        if hasattr(chunk, 'content') and chunk.content:
            markdown_output += chunk.content
        elif isinstance(chunk, str):
            markdown_output += chunk

    # --- Post-processing to fix the CSV file ---
    try:
        # Find the python code block for the data list in the agent's markdown response
        match = re.search(r"data\s*=\s*(\[.*?\])", markdown_output, re.DOTALL)
        if match:
            data_str = match.group(1)
            # Safely evaluate the string to a Python list of dictionaries
            data_list = ast.literal_eval(data_str)
            
            # Define headers as specified in the agent's prompt for consistent order
            fieldnames = [
                "Product Name", "Vendor Name", "Product Title", "Price", "Currency", 
                "Bulk Discounts or Deals", "Vendor Website", "Short Product Description", 
                "Minimum Order Quantity", "Shipping Time"
            ]
            
            # Rewrite the CSV file correctly
            with open('data.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in data_list:
                    # Write row if it's a dictionary (handles both data and empty dicts for newlines)
                    if isinstance(row, dict):
                        writer.writerow(row)
                        
    except (ValueError, SyntaxError) as e:
        # If parsing fails, the original messy CSV will be served.
        print(f"Warning: Could not parse and rewrite CSV from agent output. {e}")
    # --- End of post-processing ---

    # Check if data.csv exists
    csv_exists = os.path.exists("data.csv")
    return JSONResponse({
        "markdown": markdown_output,
        "csv_available": csv_exists
    })

@app.get("/csv")
def get_csv():
    csv_path = "data.csv"
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="data.csv")
    return JSONResponse({"error": "CSV not found"}, status_code=404)

# To run: uvicorn procurement_agent:app --reload
