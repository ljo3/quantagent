"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import numpy as np
import pandas as pd

import mcp.types as types

mcp = FastMCP("Echo Server", stateless_http=True)


def reshape_column_major(data, rows, cols):
    """
    Reshape a flat list into a matrix with given rows and cols,
    preserving MATLAB's column-major order.

    Parameters:
        data (list): Flat list of length rows*cols
        rows (int): Number of rows
        cols (int): Number of columns

    Returns:
        list of lists: Matrix with dimensions [rows][cols]
    """
    if len(data) != rows * cols:
        raise ValueError(f"Data length {len(data)} does not match rows*cols ({rows*cols}).")

    matrix = [[None] * cols for _ in range(rows)]
    for idx, val in enumerate(data):
        col = idx // rows
        row = idx % rows
        matrix[row][col] = val
    return matrix


@mcp.tool(title="Build Portfolio", description="Construct a portfolio based on given parameters.")
async def build_portfolio():
    import requests
    import json

    # Define the input parameters
    payload = {
        "nargout": 5,  # DynamicPortSim returns four outputs: Z, WPath, portPath, VPath
        "rhs": [
            100,
            200,
            10
        ]
    }

    # Define the endpoint and headers
    url = "http://20.199.27.70:9910/GBWM/buildPtfAndComputeProbs"
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Parse and print the response
    if response.status_code == 200:
        result = response.json()
    else:
        print("Error:", response.status_code, response.text)

    wgrid = result['lhs'][0]['mwdata']
    valuePtf = result['lhs'][1]['mwdata']
    portIdx = result['lhs'][2]['mwdata']
    prsk = result['lhs'][3]['mwdata']
    pret = result['lhs'][4]['mwdata']

    valuePtf = reshape_column_major(valuePtf, 118, 11)
    portIdx = reshape_column_major(portIdx, 118, 10)
    return {
        "wgrid": wgrid,
        "valuePtf": valuePtf,
        "portIdx": portIdx,
        "prsk": prsk,
        "pret": pret
    }

@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text


@mcp.resource(
    uri="greeting://{name}",
    description="Get a personalized greeting",
    name="Greeting Resource",
)
def get_greeting(
    name: str,
) -> str:
    return f"Hello, {name}!"


@mcp.prompt("")
def greet_user(
    name: str = Field(description="The name of the person to greet"),
    style: str = Field(description="The style of the greeting", default="friendly"),
) -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
