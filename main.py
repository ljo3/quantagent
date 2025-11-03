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


def _to_numpy_from_mps_array(arr):
    """
    Convert an MPS-encoded MATLAB array (dict with mwdata, mwsize, mwtype)
    to a NumPy array preserving MATLAB column-major order.
    """
    if not isinstance(arr, dict) or 'mwdata' not in arr or 'mwsize' not in arr:
        raise ValueError("Invalid MPS array format. Expected dict with 'mwdata' and 'mwsize'.")

    raw = arr['mwdata']
    # Convert "NaN" strings (common in JSON) to real np.nan
    data = [np.nan if (isinstance(x, str) and x.lower() == 'nan') else x for x in raw]

    # Determine dtype
    mwtype = arr.get('mwtype', '').lower()
    if 'double' in mwtype or 'single' in mwtype or mwtype == '':
        dtype = float
    elif 'int' in mwtype:
        # If there are NaNs, float is safer even for integer declared type
        dtype = float if any(isinstance(x, float) and np.isnan(x) for x in data) else int
    else:
        # fallback
        dtype = float

    # Shape (MATLAB uses column-major)
    # mwsize is typically like [rows, cols, ...]
    shape = tuple(int(s) for s in arr['mwsize'])
    if np.prod(shape) != len(data):
        raise ValueError(f"Size mismatch: mwsize {shape} implies {np.prod(shape)} elements "
                         f"but mwdata has {len(data)}")

    # Create numpy array with Fortran order to preserve MATLAB layout
    np_arr = np.array(data, dtype=float)  # use float to support NaNs consistently
    np_arr = np_arr.reshape(shape, order='F')
    return np_arr

def mps_lhs_to_dataframe(mps_response, column_names=None):
    """
    Convert an MPS JSON response with 'lhs' (list of MATLAB arrays encoded as dicts)
    into a pandas DataFrame. Assumes each lhs item is a column vector or something
    that can be flattened consistently to a column.

    Parameters
    ----------
    mps_response : dict
        e.g., {'lhs': [ {'mwdata': [...], 'mwsize': [11,1], 'mwtype':'double'}, ... ]}
    column_names : list[str] | None
        Optional custom column names. Must match the number of lhs items.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(mps_response, dict) or 'lhs' not in mps_response:
        raise ValueError("Expected a dict with key 'lhs'.")

    lhs = mps_response['lhs']
    if not isinstance(lhs, list) or len(lhs) == 0:
        raise ValueError("'lhs' must be a non-empty list.")

    columns = []
    nrows = None

    for i, arr in enumerate(lhs):
        np_arr = _to_numpy_from_mps_array(arr)

        # Typical case from your sample: each is [11,1], i.e., a column vector
        # Flatten in column-major to match MATLAB linearization
        col = np_arr.reshape(-1, order='F')

        if nrows is None:
            nrows = col.shape[0]
        elif col.shape[0] != nrows:
            raise ValueError(f"All columns must have the same number of rows. "
                             f"Column {i} has {col.shape[0]}, expected {nrows}.")

        columns.append(col)

    data_matrix = np.column_stack(columns)

    # Build column names
    if column_names is not None:
        if len(column_names) != data_matrix.shape[1]:
            raise ValueError("column_names length does not match number of columns.")
        cols = column_names
    else:
        cols = [f"col{i+1}" for i in range(data_matrix.shape[1])]

    df = pd.DataFrame(data_matrix, columns=cols)

    # Optional: if any columns are effectively integers (no NaN and all integral),
    # you could downcast. Here we keep float because of possible NaNs.
    return df

@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text

@mcp.tool(title="Build Portfolio", description="Construct a portfolio based on given parameters.")
def build_portfolio():
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

@mcp.tool(title="Dynamic Portfolio Simulation",description="Simulate dynamic portfolio paths based on given parameters.")
def dynamic_portfolio_simulation(wgrid, pret, prsk, portIdx, valuePtf):
    import requests
    import json

    # Define the input parameters
    payload = {
        "nargout": 4,  # DynamicPortSim returns four outputs: Z, WPath, portPath, VPath
        "rhs": [
            100,  # W0: Initial wealth
            wgrid,
            pret,
            prsk,
            portIdx,
            valuePtf,
            1  # resetFlag: Optional RNG reset
        ]
    }

    # Define the endpoint and headers
    url = "http://20.199.27.70:9910/GBWM/DynamicPortSim"
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Parse and print the response
    if response.status_code == 200:
        resultDynamicPtf = response.json()
    else:
        print("Error:", response.status_code, response.text)

    return mps_lhs_to_dataframe(resultDynamicPtf, column_names=["Z", "Wealth", "OptimumPtf", "ProbSuccess"])


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
