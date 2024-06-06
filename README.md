# Tensors-Anaylzer

## Installation

1. Create a virtual environment and install the required packages
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Install the `Tensor-Dumper` package
    ```bash
    git submodule update --init --recursive
    cd Tensor-Dumper
    pip install .
    ```

3. Install the custom C++ extension
    ```bash
    cd csrc
    bash build.sh
    ```

4. Install the `Tensors-Analyzer` package
    ```bash
    pip install .
    ```

## Test CSRC Extension

1. Run the test script
    ```bash
    python tests/test_numpy_helper.py
    ```

## Basic Usage

## Example Usage

1. Follow the `example.sh` script to analyze the tensors
    ```bash
    bash example.sh
    ```

