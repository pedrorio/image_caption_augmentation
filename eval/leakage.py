from eval.records import CaptionRecord
from eval.text import normalize_caption


class LeakageError(Exception):
    pass


def assert_image_split(
    train_images: list[CaptionRecord],
    test_images: list[CaptionRecord],
    paraphrases_by_image_id: dict[str, list[str]],
) -> None:
    """Training paraphrases may exist only for training image ids."""
    train_ids = {image.image_id for image in train_images}
    test_ids = {image.image_id for image in test_images}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise LeakageError(f"image ids in both train and test: {overlap[:5]}")
    outside = sorted(set(paraphrases_by_image_id) - train_ids)
    if outside:
        raise LeakageError(f"paraphrase for non-train image: {outside[:5]}")
    leaked_keys = sorted(set(paraphrases_by_image_id) & test_ids)
    if leaked_keys:
        raise LeakageError(f"paraphrase stored for a test image: {leaked_keys[:5]}")


def assert_no_leakage(
    train_images: list[CaptionRecord],
    test_images: list[CaptionRecord],
    paraphrases_by_image_id: dict[str, list[str]],
    training_texts: list[str],
    test_paraphrases: dict[str, list[str]] | None = None,
) -> None:
    """Fail if a test caption or a paraphrase of a test caption is in training text.

    Image ids are checked as well. Callers that use this on RSICD should know
    the released captions repeat generic sentences across images; the automated
    test uses distinct strings so a copied test sentence is unambiguous.
    """
    assert_image_split(train_images, test_images, paraphrases_by_image_id)
    forbidden = [caption for image in test_images for caption in image.captions]
    for texts in (test_paraphrases or {}).values():
        forbidden.extend(texts)
    training_norm = {normalize_caption(text) for text in training_texts}
    training_norm.discard("")
    for text in forbidden:
        normalized = normalize_caption(text)
        if normalized and normalized in training_norm:
            raise LeakageError(f"test text appears in training data: {text}")
