from typing import List, Union
from dataclasses import dataclass
from .Sentence import Sentence

@dataclass
class Image:
    sentids: List[int]
    imgid: int
    sentences: List[Sentence]
    split: Union['train', 'test']
    filename: str