"""
System monitoring utilities for tracking hardware resources.

Provides functions for monitoring RAM and GPU memory usage.
"""

import psutil
import GPUtil


def get_system_memory() -> None:
    """Print system RAM memory information."""
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024 ** 3)
    used_memory = mem.used / (1024 ** 3)
    available_memory = mem.available / (1024 ** 3)
    
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")


def get_gpu_memory() -> None:
    """Print GPU memory information."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"Total Memory: {gpu.memoryTotal} MB")
        print(f"Used Memory: {gpu.memoryUsed} MB")
        print(f"Free Memory: {gpu.memoryFree} MB")


def print_system_info() -> None:
    """Print complete system information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print("\nSystem Memory:")
    get_system_memory()
    print("\nGPU Memory:")
    get_gpu_memory()
    print("=" * 70 + "\n")
