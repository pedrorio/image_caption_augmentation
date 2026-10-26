from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CaptionRecord:
    image_id: str
    filename: str
    split: str
    captions: list[str]
    raw: dict[str, Any]
    image_path: Optional[Path] = field(default=None)
