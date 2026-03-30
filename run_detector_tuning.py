from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    STUDY_DIR,
    ARTIFACT_DIR,
    METER_CONFIG,
    FALSE_ALERT_BUDGET,
)
from src.io import load_meter_raw
from src.preprocess import add_derived_features, split_chronologically
from src.benchmark import inject_synthetic_anomalies
from src.features import fit_slot_baseline, apply_slot_baseline, add_context_features
from src.hybrid import build_hybrid_score_frame
from src.evaluation import threshold_to_events, score_event_predictions, score_point_predictions, compute_weeks
from src.selection import build_threshold_grid, choose_threshold
from src.tracking import setup_tracking, log_candidate_run, log_summary_run
from src.tuning_spaces import get_tuning_variants


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "recommended_model" in df.columns:
        df = df.rename(
            columns={
                "recommended_model": "model_name",
                "recommended_feature_set": "feature_set",
                "recommended_threshold": "selected_threshold",
            }
        )

    required = {"meter_id", "model_name", "feature_set", "selected_threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"deployment_recommendations.csv missing columns: {sorted(missing)}")

    return df


def prepare_meter_bundle(meter_id: str) -> Dict[str, pd.DataFrame]:
    clean = add_derived_features(load_meter_raw(RAW_DIR, meter_id))
    train, val_clean, test_clean = split_chronologically(clean)
    val_injected, val_events = inject_synthetic_anomalies(val_clean, meter_id, seed=42, split_name="val")
    test_injected, test_events = inject_synthetic_anomalies(test_clean, meter_id, seed=84, split_name="test")
    events = pd.concat([val_events, test_events], ignore_index=True)

    prefix = f"{meter_id}_{str(METER_CONFIG[meter_id]['meter_name']).lower().replace(' ', '_')}"
    write_df(clean, PROCESSED_DIR / f"{prefix}_clean_full.csv")
    write_df(train, PROCESSED_DIR / f"{prefix}_train.csv")
    write_df(val_injected, PROCESSED_DIR / f"{prefix}_val.csv")
    write_df(test_injected, PROCESSED_DIR / f"{prefix}_test.csv")
    write_df(events, PROCESSED_DIR / f"{prefix}_events.csv")

    return {
        "clean_full": clean,
        "train": train,
        "val": val_injected,
        "test": test_injected,
        "events": events,
    }


def choose_best_tuned_variant(df: pd.DataFrame, meter_id: str) -> pd.Series:
    budget = FALSE_ALERT_BUDGET[meter_id]
    tmp = df.copy()
    tmp["within_budget"] = tmp["val_false_alerts_per_week"] <= budget
    tmp["val_utility"] = tmp["val_event_f1"] - 0.05 * tmp["val_false_alerts_per_week"]

    feasible = tmp[tmp["within_budget"]].copy()
    if not feasible.empty:
        nonzero = feasible[feasible["val_event_recall"] > 0].copy()
        if not nonzero.empty:
            return nonzero.sort_values(
                ["val_utility", "val_event_recall", "val_event_f1", "val_false_alerts_per_week"],
                ascending=[False, False, False, True],
            ).iloc[0]

        return feasible.sort_values(
            ["val_utility", "val_event_f1", "val_false_alerts_per_week"],
            ascending=[False, False, True],
        ).iloc[0]

    nonzero = tmp[tmp["val_event_recall"] > 0].copy()
    if not nonzero.empty:
        return nonzero.sort_values(
            ["val_utility", "val_event_recall", "val_event_f1", "val_false_alerts_per_week"],
            ascending=[False, False, False, True],
        ).iloc[0]

    return tmp.sort_values(
        ["val_false_alerts_per_week", "val_event_f1"],
        ascending=[True, False],
    ).iloc[0]

def evaluate_tuning_for_meter(meter_id: str, model_family: str, feature_set: str) -> tuple[pd.DataFrame, pd.Series]:
    bundle = prepare_meter_bundle(meter_id)

    train = add_context_features(bundle["train"].copy())
    val = add_context_features(bundle["val"].copy())
    test = add_context_features(bundle["test"].copy())

    baseline, fallback = fit_slot_baseline(train)
    train_b = apply_slot_baseline(train, baseline, fallback)
    val_b = apply_slot_baseline(val, baseline, fallback)
    test_b = apply_slot_baseline(test, baseline, fallback)

    true_val_events = bundle["events"][bundle["events"]["split"] == "val"].copy()
    true_test_events = bundle["events"][bundle["events"]["split"] == "test"].copy()

    tuning_dir = STUDY_DIR / "tuning"
    threshold_dir = tuning_dir / "threshold_sweeps"
    artifact_dir = ARTIFACT_DIR / "tuning" / meter_id / model_family
    ensure_dirs(tuning_dir, threshold_dir, artifact_dir)

    rows: List[dict] = []

    variants = get_tuning_variants(meter_id, model_family)

    for variant in variants:
        variant_name = variant["name"]
        hybrid_params = variant.get("hybrid_params", {})
        event_params = variant.get("event_params", {})

        if model_family == "seasonal_residual":
            val_scored = val_b.copy()
            test_scored = test_b.copy()
            score_col = "seasonal_score"
        elif model_family == "hybrid_guarded":
            val_scored = build_hybrid_score_frame(val_b.copy(), meter_id, hybrid_params)
            test_scored = build_hybrid_score_frame(test_b.copy(), meter_id, hybrid_params)
            score_col = "hybrid_score"
        else:
            raise ValueError(f"Unsupported tuning family: {model_family}")

        curve_rows = []
        for threshold in build_threshold_grid(val_scored[score_col], model_family):
            val_points, val_events = threshold_to_events(
                val_scored.copy(), meter_id, score_col, threshold, model_family, event_params=event_params
            )
            val_point = score_point_predictions(val_points)
            val_event = score_event_predictions(val_events, true_val_events)
            val_false = val_event["false_alerts"] / compute_weeks(val_scored)

            curve_rows.append(
                {
                    "meter_id": meter_id,
                    "model_name": model_family,
                    "feature_set": feature_set,
                    "variant_name": variant_name,
                    "threshold": threshold,
                    **val_point,
                    **val_event,
                    "val_false_alerts_per_week": val_false,
                }
            )

        curve_df = pd.DataFrame(curve_rows)
        curve_path = threshold_dir / f"{meter_id}_{feature_set}_{model_family}_{variant_name}.csv"
        write_df(curve_df, curve_path)

        chosen = choose_threshold(curve_df, FALSE_ALERT_BUDGET[meter_id])
        threshold = float(chosen["threshold"])

        test_points, test_events = threshold_to_events(
            test_scored.copy(), meter_id, score_col, threshold, model_family, event_params=event_params
        )
        test_point = score_point_predictions(test_points)
        test_event = score_event_predictions(test_events, true_test_events)
        test_false = test_event["false_alerts"] / compute_weeks(test_scored)

        run_id = log_candidate_run(
            run_name=f"tuning__{meter_id}__{feature_set}__{model_family}__{variant_name}",
            params={
                "meter_id": meter_id,
                "feature_set": feature_set,
                "model_name": model_family,
                "variant_name": variant_name,
                "selected_threshold": threshold,
                "false_alert_budget_per_week": FALSE_ALERT_BUDGET[meter_id],
                "hybrid_params_json": json.dumps(hybrid_params, sort_keys=True),
                "event_params_json": json.dumps(event_params, sort_keys=True),
            },
            metrics={
                "val_point_precision": float(chosen["point_precision"]),
                "val_point_recall": float(chosen["point_recall"]),
                "val_point_f1": float(chosen["point_f1"]),
                "val_event_precision": float(chosen["event_precision"]),
                "val_event_recall": float(chosen["event_recall"]),
                "val_event_f1": float(chosen["event_f1"]),
                "val_false_alerts_per_week": float(chosen["val_false_alerts_per_week"]),
                "test_point_precision": float(test_point["point_precision"]),
                "test_point_recall": float(test_point["point_recall"]),
                "test_point_f1": float(test_point["point_f1"]),
                "test_event_precision": float(test_event["event_precision"]),
                "test_event_recall": float(test_event["event_recall"]),
                "test_event_f1": float(test_event["event_f1"]),
                "test_false_alerts_per_week": float(test_false),
            },
            artifact_paths=[curve_path],
            tags={
                "stage": "detector_tuning",
                "meter_id": meter_id,
                "feature_set": feature_set,
                "model_name": model_family,
                "variant_name": variant_name,
            },
        )

        rows.append(
            {
                "meter_id": meter_id,
                "feature_set": feature_set,
                "model_name": model_family,
                "variant_name": variant_name,
                "selected_threshold": threshold,
                "run_id": run_id,
                "val_point_precision": float(chosen["point_precision"]),
                "val_point_recall": float(chosen["point_recall"]),
                "val_point_f1": float(chosen["point_f1"]),
                "val_event_precision": float(chosen["event_precision"]),
                "val_event_recall": float(chosen["event_recall"]),
                "val_event_f1": float(chosen["event_f1"]),
                "val_false_alerts_per_week": float(chosen["val_false_alerts_per_week"]),
                "test_point_precision": float(test_point["point_precision"]),
                "test_point_recall": float(test_point["point_recall"]),
                "test_point_f1": float(test_point["point_f1"]),
                "test_event_precision": float(test_event["event_precision"]),
                "test_event_recall": float(test_event["event_recall"]),
                "test_event_f1": float(test_event["event_f1"]),
                "test_false_alerts_per_week": float(test_false),
                "hybrid_params_json": json.dumps(hybrid_params, sort_keys=True),
                "event_params_json": json.dumps(event_params, sort_keys=True),
            }
        )

    results_df = pd.DataFrame(rows)
    results_path = tuning_dir / f"{meter_id}__{model_family}__tuning_results.csv"
    write_df(results_df, results_path)

    winner = choose_best_tuned_variant(results_df, meter_id)
    return results_df, winner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter-id", choices=["m1", "m2"], default=None)
    args = parser.parse_args()

    setup_tracking("activeu_detector_tuning")

    rec_path = STUDY_DIR / "deployment_recommendations.csv"
    rec_df = normalize_recommendations(pd.read_csv(rec_path))

    if args.meter_id:
        rec_df = rec_df[rec_df["meter_id"] == args.meter_id].copy()

    recommendation_rows = []
    payload = {"params": {}, "metrics": {}}

    for _, row in rec_df.iterrows():
        meter_id = row["meter_id"]
        model_family = row["model_name"]
        feature_set = row["feature_set"]

        results_df, winner = evaluate_tuning_for_meter(meter_id, model_family, feature_set)
        recommendation_rows.append(dict(winner))

        payload["params"][f"{meter_id}_winner_model"] = winner["model_name"]
        payload["params"][f"{meter_id}_winner_variant"] = winner["variant_name"]
        payload["params"][f"{meter_id}_winner_threshold"] = winner["selected_threshold"]
        payload["metrics"][f"{meter_id}_val_event_f1"] = winner["val_event_f1"]
        payload["metrics"][f"{meter_id}_val_false_alerts_per_week"] = winner["val_false_alerts_per_week"]
        payload["metrics"][f"{meter_id}_test_event_f1"] = winner["test_event_f1"]
        payload["metrics"][f"{meter_id}_test_false_alerts_per_week"] = winner["test_false_alerts_per_week"]

    tuning_dir = STUDY_DIR / "tuning"
    ensure_dirs(tuning_dir)
    recommendation_df = pd.DataFrame(recommendation_rows)
    recommendation_path = tuning_dir / "tuning_recommendations.csv"
    write_df(recommendation_df, recommendation_path)

    log_summary_run(
        "detector_tuning_summary",
        payload,
        artifact_paths=[tuning_dir],
    )

    print("\nDetector tuning complete.")
    print(recommendation_df[["meter_id", "model_name", "variant_name", "selected_threshold"]])


if __name__ == "__main__":
    main()