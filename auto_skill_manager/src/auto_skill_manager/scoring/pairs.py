from __future__ import annotations

import heapq
import math

from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.scorecard import PairScoreCard
from auto_skill_manager.schema.skill import SkillRecord
from auto_skill_manager.scoring.anchors import anchor_strength, tokenize


def skill_text(skill: SkillRecord) -> str:
    anchors = skill.anchors or {}
    parts = [skill.name, skill.description, " ".join(skill.examples)]
    parts.extend(anchors.get("verbs", []))
    parts.extend(anchors.get("objects", []))
    parts.extend(anchors.get("constraints", []))
    return " ".join(part for part in parts if part)


def skill_tokens(skill: SkillRecord) -> set[str]:
    return tokenize(skill_text(skill))


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _term_frequencies(text: str) -> dict[str, int]:
    tf: dict[str, int] = {}
    for token in tokenize(text):
        tf[token] = tf.get(token, 0) + 1
    return tf


def _build_idf(corpus: list[dict[str, int]]) -> dict[str, float]:
    n = len(corpus)
    df: dict[str, int] = {}
    for tf in corpus:
        for term in tf:
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _tfidf_vector(tf: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
    return {term: count * idf.get(term, 1.0) for term, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = 0.0
    for term, weight in a.items():
        if term in b:
            dot += weight * b[term]
    norm_a = math.sqrt(sum(v * v for v in a.values())) if a else 0.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) if b else 0.0
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


def tfidf_similarity(text_i: str, text_j: str, idf: dict[str, float]) -> float:
    tf_i = _term_frequencies(text_i)
    tf_j = _term_frequencies(text_j)
    vec_i = _tfidf_vector(tf_i, idf)
    vec_j = _tfidf_vector(tf_j, idf)
    return _cosine_similarity(vec_i, vec_j)


def pair_scorecards(
    library: LibraryRecord,
    max_pairs: int | None = None,
    *,
    beta: float = 20.0,
) -> list[PairScoreCard]:
    results: list[PairScoreCard] = []
    heap: list[tuple[float, float, int, PairScoreCard]] = []
    counter = 0

    skill_texts = {skill.id: skill_text(skill) for skill in library.skills}
    corpus_tfs = {sid: _term_frequencies(text) for sid, text in skill_texts.items()}
    idf = _build_idf(list(corpus_tfs.values()))
    tfidf_vectors = {sid: _tfidf_vector(tf, idf) for sid, tf in corpus_tfs.items()}
    skill_strengths = {skill.id: anchor_strength(skill) for skill in library.skills}

    norm_ref = math.exp(beta * 0.5) - 1.0

    for index, left in enumerate(library.skills):
        left_vec = tfidf_vectors[left.id]
        for right in library.skills[index + 1:]:
            right_vec = tfidf_vectors[right.id]
            similarity = _cosine_similarity(left_vec, right_vec)
            same_family = left.family is not None and left.family == right.family

            shared_verbs = sorted(set(left.anchors.get("verbs", [])) & set(right.anchors.get("verbs", [])))
            shared_objects = sorted(set(left.anchors.get("objects", [])) & set(right.anchors.get("objects", [])))
            anchor_overlap = min(1.0, 0.2 * len(shared_verbs) + 0.2 * len(shared_objects))

            boltzmann_raw = math.exp(beta * similarity)
            competition_risk = min(1.0, (boltzmann_raw - 1.0) / norm_ref) if norm_ref > 0 else 0.0

            if similarity >= 0.45:
                interference_direction = "symmetric"
            elif len(shared_objects) > len(shared_verbs):
                interference_direction = "object-driven"
            else:
                interference_direction = "weak"

            overlap_score = min(1.0, 0.7 * similarity + 0.3 * anchor_overlap)
            merge_candidate_score = min(1.0, 0.75 * competition_risk + (0.1 if same_family else 0.0))
            split_signal = 0.25 if same_family and similarity < 0.22 else 0.0
            weak_drag_risk = min(1.0, 0.65 * similarity + 0.2 * anchor_overlap)

            if same_family:
                strength_gap = abs(skill_strengths[left.id] - skill_strengths[right.id])
                strong_tow = strength_gap * similarity
            else:
                strong_tow = 0.0

            card = PairScoreCard(
                left_skill_id=left.id,
                right_skill_id=right.id,
                semantic_similarity=round(similarity, 3),
                overlap_score=round(overlap_score, 3),
                competition_risk=round(competition_risk, 3),
                interference_direction=interference_direction,
                merge_candidate_score=round(merge_candidate_score, 3),
                split_signal=round(split_signal, 3),
                strong_tow_potential=round(strong_tow, 3),
                weak_drag_risk=round(weak_drag_risk, 3),
                details={
                    "same_family": same_family,
                    "shared_verbs": shared_verbs,
                    "shared_objects": shared_objects,
                    "anchor_overlap": round(anchor_overlap, 3),
                    "boltzmann_raw": boltzmann_raw,
                },
            )
            if max_pairs is None or max_pairs <= 0:
                results.append(card)
            else:
                key = (card.competition_risk, card.semantic_similarity)
                entry = (key[0], key[1], counter, card)
                counter += 1
                if len(heap) < max_pairs:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
    if max_pairs is not None and max_pairs > 0:
        results = [item[3] for item in heap]
    results.sort(key=lambda item: (item.competition_risk, item.semantic_similarity), reverse=True)
    return results
