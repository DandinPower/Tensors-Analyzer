import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from time import perf_counter

from .constants import HISTOGRAM_SIZE

def draw_all_distribution(save_name:str, tensors: np.ndarray, start_level: int, end_level: int) -> None:
    drawing_levels = np.logspace(start_level, end_level, num=HISTOGRAM_SIZE)
    
    abs_save_name = os.path.abspath(save_name)
    dir = os.path.dirname(abs_save_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(f"Drawing all distribution...")
    start = perf_counter()
    plt.title(f"All distribution")
    plt.xlabel("Value")
    plt.ylabel("Number of elements")
    plt.hist(tensors, bins=drawing_levels)
    plt.gca().set_xscale("log")  # Set x-axis to logarithmic scale

    def format_func(value, tick_number):
        return f"1e{int(np.log10(value))}"
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.savefig(save_name, dpi=300)
    plt.clf()
    end = perf_counter()
    print(f"Drawing time: {end - start}")