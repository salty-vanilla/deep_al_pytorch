import abc
from pathlib import Path
from typing import Tuple
import torch



class Dataset(object):
    def __init__(self, download_path: str) -> None:
        self.download_path = Path(download_path)

    @abc.abstractmethod
    def __call__(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        raise NotImplementedError
