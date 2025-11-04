"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import numpy as np
import pandas as pd

import mcp.types as types

mcp = FastMCP("Echo Server", stateless_http=True)


@mcp.tool(title="Build Portfolio", description="Construct a portfolio based on given parameters.")
async def build_portfolio():
    import httpx

    # Define the input parameters
    payload = {
        "nargout": 5,  # DynamicPortSim returns four outputs: Z, WPath, portPath, VPath
        "rhs": [100, 200, 10]
    }

    # Define the endpoint and headers
    url = "http://20.199.27.70:9910/GBWM/buildPtfAndComputeProbs"
    headers = {"Content-Type": "application/json"}

    # Send the POST request using httpx
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, headers=headers, json=payload)

    # Parse and handle the response
    if response.status_code == 200:
        result = response.json()
        wgrid = result['lhs'][0]['mwdata']
        valuePtf = result['lhs'][1]['mwdata']
        portIdx = result['lhs'][2]['mwdata']
        prsk = result['lhs'][3]['mwdata']
        pret = result['lhs'][4]['mwdata']

        # Return structured data instead of a static string
        return {
            "wgrid": wgrid,
            "valuePtf": valuePtf,
            "portIdx": portIdx,
            "prsk": prsk,
            "pret": pret
        }
    else:
        return {"error": f"Status {response.status_code}: {response.text}"}


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
