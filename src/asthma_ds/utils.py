def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.
    Creates it if missing.
    """
    path.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Path, msg: str = None) -> None:
    """
    Save DataFrame to Parquet with optional confirmation message.
    """
    df.to_parquet(path, index=False)
    if msg:
        print(f"✅ Saved {msg} to {path}")
    else:
        print(f"✅ Saved to {path}")


def save_csv(df: pd.DataFrame, path: Path, msg: str = None) -> None:
    """
    Save DataFrame to CSV with optional confirmation message.
    """
    df.to_csv(path, index=False)
    if msg:
        print(f"✅ Saved {msg} to {path}")
    else:
        print(f"✅ Saved to {path}")


def describe_distribution(series: pd.Series) -> dict:
    """
    Quick descriptive stats for a numeric series.
    """
    return {
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "std": series.std(),
        "q1": series.quantile(0.25),
        "median": series.median(),
        "q3": series.quantile(0.75),
    }
