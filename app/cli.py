"""CLI entry point: run the eval harness and print a report table."""
from __future__ import annotations

import argparse

from app.harness import run_default_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation harness.")
    parser.add_argument("--k", type=int, default=5, help="cutoff for precision@k / recall@k")
    args = parser.parse_args()

    report = run_default_eval(k=args.k)

    print(f"Retrieval Eval Report (k={report.k}, queries={report.num_queries})")
    print("-" * 72)
    print(f"{'query_id':<10}{'precision@k':<14}{'recall@k':<12}{'MRR':<8}")
    for q in report.per_query:
        print(f"{q.query_id:<10}{q.precision_at_k:<14.2f}{q.recall_at_k:<12.2f}{q.reciprocal_rank:<8.2f}")
    print("-" * 72)
    print(
        f"mean precision@k={report.mean_precision_at_k:.3f}  "
        f"mean recall@k={report.mean_recall_at_k:.3f}  "
        f"mean MRR={report.mean_reciprocal_rank:.3f}"
    )


if __name__ == "__main__":
    main()
