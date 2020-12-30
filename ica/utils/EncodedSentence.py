from typing import List, Union
from dataclasses import dataclass
from torch.tensor import Tensor

@dataclass
class EncodedSentence:
    x_inputs: Tensor
    x_attention: Tensor
    y_inputs: Tensor
    y_attention: Tensor