from typing import List, Union
from dataclasses import dataclass


@dataclass
class Sentence:
    tokens: List[str]
    raw: str
    imgid: int
    sentid: int