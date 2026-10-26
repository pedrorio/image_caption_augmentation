import argparse
import json

from eval.config import load_eval_config
from eval.data import group_by_official_split, load_caption_records
from eval.leakage import assert_image_split
from eval.pipeline import run_eval


def _leakage_only(path: str) -> dict:
    cfg = load_eval_config(path)
    records = load_caption_records(cfg.captions_json)
    groups = group_by_official_split(
        records,
        max_images=cfg.max_images,
        seed=cfg.seed,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        min_captions_per_image=cfg.min_captions_per_image,
    )
    assert_image_split(groups["train"], groups["test"], {})
    summary = {
        "scale": cfg.scale,
        "n_train_images": len(groups["train"]),
        "n_val_images": len(groups["val"]),
        "n_test_images": len(groups["test"]),
        "train_test_image_overlap": 0,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Paraphrase quality and captioner A/B.")
    parser.add_argument("command", choices=("all", "leakage", "ab"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "leakage":
        _leakage_only(args.config)
        return
    if args.command == "all" or args.command == "ab":
        report = run_eval(load_eval_config(args.config))
        decision = report["downstream"]["decision"]
        print(
            f"scale={report['scale']} label={report['label']} "
            f"cider_mean_diff={decision['mean_diff']:.6f} action={decision['action']}"
        )
        print(f"wrote {load_eval_config(args.config).output_json}")
        return
    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
