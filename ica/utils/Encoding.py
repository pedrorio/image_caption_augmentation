from dataclasses import dataclass

from torch import Tensor


@dataclass
class Encoding:
    input_ids: Tensor
    attention_mask: Tensor
