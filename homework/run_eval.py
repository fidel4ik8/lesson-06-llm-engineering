"""Batch-evaluator: runs extraction_agent on all samples × providers,
saves per-run JSON results and aggregated CSV."""
import csv
import json
import os
from pathlib import Path

from extraction_agent import extract_meeting_data

HW = Path(__file__).parent
SAMPLES = HW / "samples"
RESULTS = HW / "results"
RESULTS.mkdir(exist_ok=True)

# Ground truth — counted manually from sample texts
GROUND_TRUTH = {
    "simple": {
        "tasks_expected": 4,   # Іван UI, Мартин API, Олена frontend, Анна ТЗ
        "people": ["Анна", "Іван", "Мартин", "Олена"],
    },
    "chaotic": {
        "tasks_expected": 6,   # Костя нотифікації, Костя payment, Маша login, Дімон експорт, Дімон синк з Олегом, Маша міграція
        "people": ["Костя", "Маша", "Дімон", "Олег"],
    },
    "technical": {
        "tasks_expected": 5,   # Артем Terraform, Юлія search-сервіс, Юлія rate-limiter, Богдан indexing, Богдан load-test
        "people": ["Сергій", "Юлія", "Артем", "Богдан"],
    },
}

DATASETS = ["simple", "chaotic", "technical"]
PROVIDERS = ["ollama", "openai"]
FILE_MAP = {
    "simple": "simple_meeting.txt",
    "chaotic": "chaotic_standup.txt",
    "technical": "technical_sync.txt",
}


def count_hallucinations(data: dict, allowed_people: list[str]) -> int:
    """Crude: count tasks whose owner is not among allowed people."""
    if not data or "tasks" not in data:
        return 0
    halluc = 0
    for t in data["tasks"]:
        owner = (t.get("owner") or "").strip()
        if not owner:
            continue
        if not any(p.lower() in owner.lower() or owner.lower() in p.lower() for p in allowed_people):
            halluc += 1
    return halluc


def main():
    csv_path = HW / "eval_results.csv"
    rows = []

    for dataset in DATASETS:
        text = (SAMPLES / FILE_MAP[dataset]).read_text(encoding="utf-8")
        gt = GROUND_TRUTH[dataset]
        for provider in PROVIDERS:
            print(f"\n=== {dataset} / {provider} ===")
            try:
                data, metrics = extract_meeting_data(text, provider=provider)
            except Exception as e:
                print(f"  ERROR: {e}")
                rows.append({
                    "dataset": dataset, "provider": provider,
                    "json_valid": False, "tasks_found": 0,
                    "tasks_expected": gt["tasks_expected"],
                    "hallucinations": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cost_usd": 0, "latency_sec": 0, "error": str(e)[:200],
                })
                continue

            tasks_found = len(data.get("tasks", [])) if data else 0
            halluc = count_hallucinations(data, gt["people"])

            # Save per-run artifact
            out_file = RESULTS / f"{dataset}_{provider}.json"
            payload = {
                "metrics": {k: v for k, v in metrics.items() if k != "raw_response"},
                "data": data,
                "raw_response": metrics["raw_response"] if not metrics["json_valid"] else None,
            }
            out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"  json_valid={metrics['json_valid']} "
                  f"tasks={tasks_found}/{gt['tasks_expected']} "
                  f"halluc={halluc} "
                  f"tokens={metrics['total_tokens']} "
                  f"cost=${metrics['cost_usd']} "
                  f"latency={metrics['latency_sec']}s")

            rows.append({
                "dataset": dataset,
                "provider": provider,
                "json_valid": metrics["json_valid"],
                "tasks_found": tasks_found,
                "tasks_expected": gt["tasks_expected"],
                "hallucinations": halluc,
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "total_tokens": metrics["total_tokens"],
                "cost_usd": metrics["cost_usd"],
                "latency_sec": metrics["latency_sec"],
                "error": "",
            })

    fieldnames = ["dataset", "provider", "json_valid", "tasks_found", "tasks_expected",
                  "hallucinations", "input_tokens", "output_tokens", "total_tokens",
                  "cost_usd", "latency_sec", "error"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV saved: {csv_path}")
    print(f"✅ Per-run JSONs in: {RESULTS}")


if __name__ == "__main__":
    main()
