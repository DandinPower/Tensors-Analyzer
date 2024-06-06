import os
import numpy as np
from tqdm import tqdm
import numpy_helper
from dumper import Dumper
from .drawer import draw_all_distribution

class Tensor:
    def __init__(self, filename: str, tensor_data: np.ndarray, start_index: int, offset: int) -> None:
        self.filename: str = filename
        self.tensor_data: np.ndarray = tensor_data
        self.start_index: int = start_index # start index in the flat tensor
        assert offset == tensor_data.size, 'Offset must be equal to the tensor size'
        self.offset: int = offset
        
    def __repr__(self) -> str:
        return f'Tensor(filename={self.filename}, tensor_shape={self.tensor_data.shape}, start_index={self.start_index}, offset={self.offset})'

    def __len__(self) -> int:
        return self.tensor_data.size
    
class TensorsAnalyzer:
    def __init__(self, verbose=False) -> None:
        self.verbose: bool = verbose
        self.tensors: list[Tensor] = []
        self.flat_tensor: np.ndarray | None = None
        self.flat_tensor_size: int = 0

    def print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def reset(self) -> None:
        self.tensors.clear()
        self.flat_tensor = None
        self.flat_tensor_size = 0

    def _parse_folder_tensors_total_elements(self, foldername: str) -> int:
        self.print(f'Parsing total elements from folder: {foldername}')
        assert os.path.exists(foldername), 'Folder not found'
        total_elements = 0
        files = os.listdir(foldername)
        for file in tqdm(files, total=len(files)):
            assert file.endswith('.npz'), 'Only .npz files are supported, please ensure all files in the folder are dumped by TensorDumper'
            temp_tensor = Dumper.load_tensor_in_numpy(os.path.join(foldername, file))
            total_elements += temp_tensor.size
        return total_elements

    def load_tensors_by_multiple_folder(self, foldername_list: list[str], is_abs=True) -> None:
        assert len(self.tensors) == 0, 'Please reset the analyzer before adding new tensors'
        assert self.flat_tensor is None, 'Please reset the analyzer before adding new tensors'
        assert self.flat_tensor_size == 0, 'Please reset the analyzer before adding new tensors'
        for foldername in foldername_list:
            assert os.path.exists(foldername), f'Folder: {foldername} not found'
            self.flat_tensor_size += self._parse_folder_tensors_total_elements(foldername)
        
        self.flat_tensor = np.zeros(self.flat_tensor_size, dtype=np.float32)
        start_index = 0
        for foldername in foldername_list:
            self.print(f'Loading tensors from folder: {foldername}')
            files = os.listdir(foldername)
            for file in tqdm(files, total=len(files)):
                assert file.endswith('.npz'), 'Only .npz files are supported, please ensure all files in the folder are dumped by TensorDumper'
                temp_tensor = Dumper.load_tensor_in_numpy(os.path.join(foldername, file))
                assert temp_tensor.dtype == np.float16 or temp_tensor.dtype == np.float32, 'Only float16 and float32 dtypes are supported'
                # turn the dtype into float32 if it is not
                if temp_tensor.dtype == np.float16:
                    temp_tensor = temp_tensor.astype(np.float32)
                # turn the tensor into a 1D array if it is not
                temp_tensor = temp_tensor.flatten()
                temp_length = temp_tensor.size
                narrow_tensor = self.flat_tensor[start_index:start_index + temp_length]

                numpy_helper.copy_src_tensor_into_dst_tensor(narrow_tensor, temp_tensor)
                if is_abs:
                    numpy_helper.tensor_abs(narrow_tensor)
                del temp_tensor
                self.tensors.append(Tensor(file, narrow_tensor, start_index, temp_length))
                start_index += temp_length

    def draw_distribution(self, save_name: str, start_level: int, end_level: int) -> None:
        assert self.flat_tensor is not None, 'Please load the tensors first'
        dirname = os.path.dirname(save_name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        draw_all_distribution(save_name, self.flat_tensor, start_level, end_level)