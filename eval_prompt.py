import argparse
import json
from pathlib import Path
from evaluate import evaluate, read_rows
from label import DATA_DIR, open_text

def load_predicted(folder: Path) -> dict:
    predicted = {}
    with open_text(folder / "predictions.jsonl") as file:
        for line in file:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get("error"):
                print(f"  [warn] Photo {row['photo_id']} error: {row['error']}")
                predicted[row["photo_id"]] = []  # Считаем, что модель ничего не нашла
            else:
                predicted[row["photo_id"]] = row["boxes"]
    return predicted

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold = read_rows(DATA_DIR / "gold/ours_gold.jsonl")
    predicted = load_predicted(args.folder)
    missing = gold.keys() - predicted.keys()
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} gold photos")

    predicted = {pid: predicted[pid] for pid in gold}
    metrics = evaluate(gold, predicted)["metrics"]
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {args.output} | F1@0.5: {metrics['0.5']['f1']:.4f}")

if __name__ == "__main__":
    main()