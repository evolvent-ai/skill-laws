from __future__ import annotations

from auto_skill_manager.ingest.loaders import to_plain_data
from auto_skill_manager.schema.optimization import OptimizationResult


def build_optimization_report_payload(result: OptimizationResult) -> dict:
    payload = to_plain_data(result)
    summary_delta = payload.get("summary_delta", {})
    skill_deltas = payload.get("skill_deltas", [])
    actions_applied = payload.get("actions_applied", [])
    actions_skipped = payload.get("actions_skipped", [])
    action_impacts = payload.get("action_impacts", [])
    scorecard_before = payload.get("scorecard_before") or {}
    scorecard_after = payload.get("scorecard_after") or {}

    payload["version"] = "0.1"
    payload["report_type"] = "auto_skill_manager_optimization"
    payload["highlights"] = {
        "plan_id": payload.get("plan_id"),
        "library_id_before": payload.get("library_id_before"),
        "library_id_after": payload.get("library_id_after"),
        "actions_applied": len(actions_applied),
        "actions_skipped": len(actions_skipped),
        "changed_skill_count": sum(1 for item in skill_deltas if item.get("changed")),
        "improved_metrics": [name for name, item in summary_delta.items() if item.get("direction") == "improved"],
        "degraded_metrics": [name for name, item in summary_delta.items() if item.get("direction") == "degraded"],
        "is_optimization": True,
    }
    payload["views"] = {
        "metric_delta_cards": [
            {
                "metric": name,
                "before": item.get("before"),
                "after": item.get("after"),
                "delta": item.get("delta"),
                "direction": item.get("direction"),
            }
            for name, item in summary_delta.items()
        ],
        "skill_delta_table": [
            {
                "skill_id": item.get("skill_id"),
                "changed": item.get("changed"),
                "competition_score_before": item.get("competition_score_before"),
                "competition_score_after": item.get("competition_score_after"),
                "competition_score_delta": round((item.get("competition_score_after") or 0.0) - (item.get("competition_score_before") or 0.0), 3),
                "anchor_strength_before": item.get("anchor_strength_before"),
                "anchor_strength_after": item.get("anchor_strength_after"),
                "anchor_strength_delta": round((item.get("anchor_strength_after") or 0.0) - (item.get("anchor_strength_before") or 0.0), 3),
                "black_hole_risk_before": item.get("black_hole_risk_before"),
                "black_hole_risk_after": item.get("black_hole_risk_after"),
                "black_hole_risk_delta": round((item.get("black_hole_risk_after") or 0.0) - (item.get("black_hole_risk_before") or 0.0), 3),
                "rewrite_priority_before": item.get("rewrite_priority_before"),
                "rewrite_priority_after": item.get("rewrite_priority_after"),
                "rewrite_priority_delta": round((item.get("rewrite_priority_after") or 0.0) - (item.get("rewrite_priority_before") or 0.0), 3),
            }
            for item in skill_deltas
        ],
        "action_summary": {
            "applied": actions_applied,
            "skipped": actions_skipped,
        },
        "action_impact_cards": [
            {
                "action_id": item.get("action_id"),
                "action_type": item.get("action_type"),
                "action_subtype": item.get("action_subtype"),
                "status": item.get("status"),
                "target_skill_ids": item.get("target_skill_ids", []),
                "changed_skill_ids": item.get("changed_skill_ids", []),
                "metric_impacts": item.get("metric_impacts", {}),
                "notes": item.get("notes", ""),
            }
            for item in action_impacts
        ],
    }
    payload["sections"] = {
        "scorecard_before": scorecard_before,
        "scorecard_after": scorecard_after,
        "summary_delta": summary_delta,
        "skill_deltas": skill_deltas,
        "actions_applied": actions_applied,
        "actions_skipped": actions_skipped,
        "action_impacts": action_impacts,
    }
    return payload
