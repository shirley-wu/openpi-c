from typing import Union, Tuple

from .collate import collate
from .openpi import OpenPIDataset, OpenPIDatasetConcatStates, CONSTRUCTOR_CLASSES
from .openpi_econd import OpenPIECond
from .openpi_emem import BaselineEntityStateConstructor, OpenPIEMem


def load_and_cache_examples(data_type: str, tokenizer, file_path: str, block_size: Union[int, Tuple[int]]):
    data_class = {
        "baseline": OpenPIDataset,
        "concat-states": OpenPIDatasetConcatStates,
        "econd": OpenPIECond,
        "emem": OpenPIEMem,
    }[data_type]
    return data_class(tokenizer=tokenizer,
                      file_path=file_path,
                      block_size=block_size)
