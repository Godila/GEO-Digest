"""Storage Backends."""
from .base import StorageBackend
from .jsonl_backend import JsonlStorage

def get_storage(data_dir="/app/data"):
    return JsonlStorage(data_dir=data_dir)

__all__ = ["StorageBackend", "JsonlStorage", "get_storage"]
