"""Pure split-size helper shared by the paraphraser datamodule."""


def split_sizes(
    n: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
) -> tuple[int, int, int]:
    """Return train, validation, and test counts that sum to ``n``.

    Fractions match the historical paraphraser split (about 70/20/10).
    The split is over caption pairs, not images.
    """
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    if n_train > n:
        n_train = n
    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val
    return n_train, n_val, n_test
