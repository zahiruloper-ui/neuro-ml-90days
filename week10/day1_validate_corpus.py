from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path(__file__).parent / "neuro_abstracts.csv"
    dataframe = pd.read_csv(input_path)

    required_columns = {"title", "abstract", "source_url"}
    missing_columns = required_columns - set(dataframe.columns)

    dataframe["abstract_word_count"] = (
        dataframe["abstract"]
        .fillna("")
        .str.split()
        .str.len()
    )

    checks = {
        "At least 20 abstracts": len(dataframe) >= 20,
        "Required columns present": not missing_columns,
        "No missing titles": not dataframe["title"].isna().any(),
        "No missing abstracts": not dataframe["abstract"].isna().any(),
        "No duplicate titles": not dataframe["title"].duplicated().any(),
        "Every abstract has at least 20 words": (
            dataframe["abstract_word_count"].min() >= 20
        ),
    }

    print("Corpus quality checks:")
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check_name}")

    if missing_columns:
        print(f"\nMissing columns: {sorted(missing_columns)}")

    print(f"\nTotal abstracts: {len(dataframe)}")
    print(f"Shortest abstract: {dataframe['abstract_word_count'].min()} words")
    print(f"Longest abstract: {dataframe['abstract_word_count'].max()} words")

    if all(checks.values()):
        print("\nCorpus is ready for embedding.")
    else:
        print("\nCorpus needs correction before embedding.")


if __name__ == "__main__":
    main()