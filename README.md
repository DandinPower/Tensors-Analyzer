## Tensors-Analyzer: A Simple Tool for Analyzing Tensors

This repository provides a tool for analyzing tensors, specifically focusing on visualizing their distribution. It utilizes numpy and matplotlib for data processing and visualization. It also writes a C++ extension to enable efficient numpy array operations, to prevent any additional memory overhead, So it can handle a large number of tensors. such as Loading, Processing, and Drawing the distribution of whole LLMs tensors.

## Installation

1. **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Tensor-Dumper (Submodule):**
    ```bash
    git submodule update --init --recursive
    cd Tensor-Dumper
    pip install .
    ```

4. **Build C++ Extension (csrc):**
    ```bash
    cd csrc
    bash build.sh
    ```

5. **Install Tensors-Analyzer:**
    ```bash
    pip install .
    ```

## Testing the C++ Extension

1. **Run Test Script:**
    ```bash
    python tests/test_numpy_helper.py
    ```

## Basic Usage

1. **Dump some Tensors into folder**
    ```python
    from dumper import Dumper 
    tensor_0 = np.random.randn(10, 10)
    tensor_2 = np.random.randn(10, 10)
    Dumper.save_tensor("path/to/folder1/0.npz", tensor_0)
    Dumper.save_tensor("path/to/folder2/1.npz", tensor_1)
    ```

1. **Import the Necessary Class:**
    ```python
    from tensors_analyzer import TensorsAnalyzer
    ```

2. **Create an Instance of the Analyzer:**
    ```python
    tensors_analyzer = TensorsAnalyzer(verbose=True) # Set verbose to True for debugging messages
    ```

3. **Load Tensors from Multiple Folders:**
    ```python
    folder_list = ['/path/to/folder1', '/path/to/folder2', ...]
    tensors_analyzer.load_tensors_by_multiple_folder(folder_list, is_abs=True) # Set is_abs to False if the folder paths are relative
    ```

4. **Draw the Distribution of the Loaded Tensors:**
    ```python
    save_name = 'distribution.png'
    start_level = -10
    end_level = 2
    tensors_analyzer.draw_distribution(save_name, start_level, end_level)
    ```

## Example Usage

The `example.sh` script demonstrates a typical workflow:

```bash
bash example.sh
```

**The script performs the following steps:**

1. Defines the folder paths containing the tensors, save filename, and histogram level ranges.
2. Executes the `example.py` script with the provided arguments.
3. Loads tensors from specified folders.
4. Draws the distribution of the tensors and saves it to the specified file.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
