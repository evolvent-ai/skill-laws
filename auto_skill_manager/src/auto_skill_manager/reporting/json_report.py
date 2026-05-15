from __future__ import annotations

import json

from auto_skill_manager.analyze.library import AnalysisResult
from auto_skill_manager.schema.optimization import LibraryDiffResult, OptimizationResult

from .optimization_report import build_optimization_report_payload
from auto_skill_manager.ingest.loaders import to_plain_data


def build_report_payload(result: AnalysisResult) -> dict:
    payload = to_plain_data(result)
    skill_cards = payload.get("skill_scorecards", [])
    pair_cards = payload.get("pair_scorecards", [])
    recommendations = payload.get("recommendations", [])
    summary = payload.get("summary", {})
    library = payload.get("library_scorecard", {})
    library_details = library.get("details", {})
    is_skill_inspection = library_details.get("view") == "inspect_skill"
    focus_skill_id = library_details.get("focus_skill_id")

    payload["version"] = "0.1"
    payload["report_type"] = "auto_skill_manager_skill_inspection" if is_skill_inspection else "auto_skill_manager_analysis"
    payload["highlights"] = {
        "health_status": summary.get("health_status", "unknown"),
        "top_risky_skill_ids": summary.get("top_risky_skill_ids", []),
        "recommendation_counts": summary.get("recommendation_counts", {}),
        "focus_skill_id": focus_skill_id,
        "is_skill_inspection": is_skill_inspection,
    }
    payload["views"] = {
        "risk_table": [
            {
                "skill_id": card.get("skill_id"),
                "competition_score": card.get("competition_score"),
                "anchor_strength_score": card.get("anchor_strength_score"),
                "black_hole_risk": card.get("black_hole_risk"),
                "rewrite_priority": card.get("rewrite_priority"),
                "family": card.get("details", {}).get("family"),
            }
            for card in skill_cards
        ],
        "conflict_queue": [
            {
                "left_skill_id": card.get("left_skill_id"),
                "right_skill_id": card.get("right_skill_id"),
                "competition_risk": card.get("competition_risk"),
                "overlap_score": card.get("overlap_score"),
                "interference_direction": card.get("interference_direction"),
            }
            for card in pair_cards[:20]
        ],
        "recommendation_queue": [
            {
                "id": rec.get("id"),
                "type": rec.get("type"),
                "priority": rec.get("priority"),
                "confidence": rec.get("confidence"),
                "summary": rec.get("summary"),
                "skill_ids": rec.get("entities", {}).get("skill_ids", []),
                "rationale_signals": rec.get("rationale", {}).get("observed_signals", []),
                "mapped_action": rec.get("suggested_action", {}).get("kind"),
                "mapped_action_notes": rec.get("suggested_action", {}).get("notes"),
            }
            for rec in recommendations
        ],
        "skill_focus": {
            "skill_id": focus_skill_id,
            "related_conflict_count": len(pair_cards),
            "related_recommendation_count": len(recommendations),
        } if is_skill_inspection else None,
    }
    payload["sections"] = {
        "library": library,
        "skills": skill_cards,
        "pairs": pair_cards,
        "recommendations": recommendations,
        "change_impact": payload.get("change_impact"),
        "summary": summary,
    }
    return payload


def render_json_report(result: AnalysisResult) -> str:
    payload = build_report_payload(result)
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def render_optimization_json_report(result: OptimizationResult) -> str:
    payload = build_optimization_report_payload(result)
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def render_diff_json_report(result: LibraryDiffResult) -> str:
    payload = to_plain_data(result)
    payload["version"] = "0.1"
    payload["report_type"] = "auto_skill_manager_library_diff"
    payload["highlights"] = {
        "base_library_id": payload.get("base_library_id"),
        "candidate_library_id": payload.get("candidate_library_id"),
        "added_skill_count": len(payload.get("added_skill_ids", [])),
        "removed_skill_count": len(payload.get("removed_skill_ids", [])),
        "changed_skill_count": sum(1 for item in payload.get("skill_deltas", []) if item.get("changed")),
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
            for name, item in (payload.get("summary_delta") or {}).items()
        ],
        "skill_delta_table": [
            {
                "skill_id": item.get("skill_id"),
                "competition_score_delta": round((item.get("competition_score_after") or 0) - (item.get("competition_score_before") or 0), 3),
                "anchor_strength_delta": round((item.get("anchor_strength_after") or 0) - (item.get("anchor_strength_before") or 0), 3),
                "black_hole_risk_delta": round((item.get("black_hole_risk_after") or 0) - (item.get("black_hole_risk_before") or 0), 3),
                "rewrite_priority_delta": round((item.get("rewrite_priority_after") or 0) - (item.get("rewrite_priority_before") or 0), 3),
                "changed": item.get("changed", False),
                "status": "added" if item.get("skill_id") in payload.get("added_skill_ids", []) else "removed" if item.get("skill_id") in payload.get("removed_skill_ids", []) else "changed" if item.get("changed") else "stable",
            }
            for item in payload.get("skill_deltas", [])
        ],
    }
    payload["sections"] = {
        "summary_delta": payload.get("summary_delta", {}),
        "skill_deltas": payload.get("skill_deltas", []),
        "added_skill_ids": payload.get("added_skill_ids", []),
        "removed_skill_ids": payload.get("removed_skill_ids", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
