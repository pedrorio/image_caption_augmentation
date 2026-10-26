import pytest

from eval.data import group_by_official_split, load_caption_records
from eval.leakage import LeakageError, assert_image_split, assert_no_leakage
from eval.records import CaptionRecord


def _record(image_id: str, split: str, caption: str) -> CaptionRecord:
    return CaptionRecord(
        image_id=image_id,
        filename=f"{image_id}.jpg",
        split=split,
        captions=[caption],
        raw={"filename": f"{image_id}.jpg", "split": split, "sentences": [{"raw": caption}]},
    )


def test_clean_split_passes():
    train = [_record("train-a", "train", "unique training caption about farmland")]
    test = [_record("test-a", "test", "unique test caption about a runway")]
    paraphrases = {"train-a": ["another wording of the farmland caption"]}
    training_texts = train[0].captions + paraphrases["train-a"]
    assert_no_leakage(train, test, paraphrases, training_texts)


def test_inserted_test_caption_fails():
    train = [_record("train-a", "train", "unique training caption about farmland")]
    test = [_record("test-a", "test", "unique test caption about a runway")]
    paraphrases = {"train-a": ["another wording of the farmland caption"]}
    training_texts = train[0].captions + paraphrases["train-a"] + test[0].captions
    with pytest.raises(LeakageError, match="test text"):
        assert_no_leakage(train, test, paraphrases, training_texts)


def test_inserted_test_paraphrase_fails():
    train = [_record("train-a", "train", "unique training caption about farmland")]
    test = [_record("test-a", "test", "unique test caption about a runway")]
    paraphrases = {"train-a": ["another wording of the farmland caption"]}
    leaked = "a paraphrase that only the test runway caption could have produced"
    training_texts = train[0].captions + paraphrases["train-a"] + [leaked]
    with pytest.raises(LeakageError, match="test text"):
        assert_no_leakage(
            train,
            test,
            paraphrases,
            training_texts,
            test_paraphrases={"test-a": [leaked]},
        )


def test_paraphrase_of_test_image_fails_id_check():
    train = [_record("train-a", "train", "unique training caption about farmland")]
    test = [_record("test-a", "test", "unique test caption about a runway")]
    with pytest.raises(LeakageError, match="non-train"):
        assert_image_split(train, test, {"test-a": ["should not exist"]})


def test_real_split_happens_before_augmentation_ids_are_disjoint():
    records = load_caption_records("data/raw/dataset_rsicd_modified.json")
    groups = group_by_official_split(
        records,
        max_images=12,
        seed=0,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        min_captions_per_image=2,
    )
    assert groups["train"] and groups["val"] and groups["test"]
    train_ids = {record.image_id for record in groups["train"]}
    test_ids = {record.image_id for record in groups["test"]}
    assert train_ids.isdisjoint(test_ids)
    assert_image_split(groups["train"], groups["test"], {})
    for record in groups["train"] + groups["val"] + groups["test"]:
        assert len(record.captions) >= 2
