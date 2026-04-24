import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parents[2] / "data"


def find_csv(data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    csvs = [p for p in data_dir.glob("*.csv") if p.name != "cleaned.csv"]
    if not csvs:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")
    if len(csvs) > 1:
        print(f"Multiple CSVs found; loading {csvs[0].name}")
    return csvs[0]


def load(path: pathlib.Path | None = None) -> pd.DataFrame:
    if path is None:
        path = find_csv()
    return pd.read_csv(path)


def report(df: pd.DataFrame) -> None:
    print("=== Shape ===")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns\n")

    print("=== Columns & dtypes ===")
    print(df.dtypes.to_string())
    print()

    print("=== Summary statistics (numeric) ===")
    stats = df.describe().loc[["mean", "std", "min", "max"]]
    print(stats.to_string())
    print()

    print("=== Missing values ===")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    mv = pd.DataFrame({"count": missing, "pct": pct})
    mv = mv[mv["count"] > 0]
    if mv.empty:
        print("No missing values.")
    else:
        print(mv.to_string())
    print()


if __name__ == "__main__":
    df = load()
    report(df)
