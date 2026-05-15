

from __future__ import annotations
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

SKILLS_DIR = SKILLS_DIR
ANALYSIS_RECORD_LIMIT = int(os.environ.get("SKILL_LAW_ANALYSIS_RECORD_LIMIT", "0"))
EMB_CACHE = data_path("processed", "bge_skill_embeddings.npz")
F01_LOG = finding_data_path("F01", "data", "gpt-5.4-mini_raw_llm_queries.jsonl")
F05_STRESS_LOG = finding_data_path("F05", "data", "raw_llm_queries_gpt-5.4-mini.jsonl")
F05_REFERENCE_SUMMARY = finding_data_path("F05", "data", "F05_summary.json")
F05_REFERENCE_ULTIMATE = finding_data_path("F05", "data", "ULTIMATE_COMBINATION_LAW.json")
OLD_CI_JSON = finding_data_path("F02", "data", "mechanism", "thermodynamic_F05_results.json")
OUT_JSON = finding_path("F05", "data", "mechanism", "local_competition_results.json")
OUT_FIG = finding_path("F05", "figures", "local_competition.png")

BETAS = [10, 20, 30]
TOPK_VALUES = [2, 3, 5]
BAND_LOW_VALUES = np.arange(0.50, 0.801, 0.05)
BAND_DELTAS = [0.05, 0.10, 0.15, 0.20]
FIXED_N_MIN_COUNT = 150
BEST_FIXED_N_COUNT = 4
EPS = 1e-12


def point_biserial(values, labels):
    arr = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(arr) < 3 or len(np.unique(arr)) < 2 or len(np.unique(y)) < 2:
        return {'r': None, 'p': None}
    r, p = stats.pointbiserialr(y, arr)
    return {'r': float(r), 'p': float(p)}



def spearman(values, labels):
    arr = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(arr) < 3 or len(np.unique(arr)) < 2 or len(np.unique(y)) < 2:
        return {'rho': None, 'p': None}
    rho, p = stats.spearmanr(arr, y)
    return {'rho': float(rho), 'p': float(p)}



def effect_size_from_threshold(values, labels, threshold):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=float)
    low = labels[values < threshold]
    high = labels[values >= threshold]
    if len(low) < 30 or len(high) < 30:
        return None
    gap = float(np.mean(high) - np.mean(low))
    bad_gap = float(np.mean(low) - np.mean(high))
    t, p = stats.ttest_ind(low, high, equal_var=False)
    return {
        'threshold': float(threshold),
        'acc_below': float(np.mean(low)),
        'acc_above': float(np.mean(high)),
        'gap_above_minus_below': gap,
        'gap_below_minus_above': bad_gap,
        't': float(t),
        'p': float(p),
        'n_below': int(len(low)),
        'n_above': int(len(high)),
    }



def decile_curve(records, metric):
    ordered = sorted(records, key=lambda r: r[metric])
    n_bins = 10
    bin_size = max(1, len(ordered) // n_bins)
    bins = []
    for i in range(n_bins):
        chunk = ordered[i * bin_size: (i + 1) * bin_size] if i < n_bins - 1 else ordered[i * bin_size:]
        if not chunk:
            continue
        bins.append({
            'bin': i + 1,
            'metric': float(np.mean([r[metric] for r in chunk])),
            'acc': float(np.mean([r['is_correct'] for r in chunk])),
            'N': float(np.mean([r['N'] for r in chunk])),
            'n': int(len(chunk)),
        })
    return bins



def topk_local_ci(sorted_sims, beta, k):
    return float(np.sum(np.exp(beta * np.asarray(sorted_sims[: min(k, len(sorted_sims))], dtype=float))))



def parse_library(system_prompt, skill2idx):
    library = []
    for line in system_prompt.split('\n'):
        if line.startswith('- ') and ':' in line:
            tid = line[2:].split(':')[0].strip()
            if tid in skill2idx:
                library.append(tid)
    return library



def load_embeddings():
    valid_skills = sorted([
        n for n in os.listdir(SKILLS_DIR)
        if os.path.exists(SKILLS_DIR / n / 'SKILL.md')
    ])
    embs = np.load(EMB_CACHE)['embs']
    n_embs = min(len(valid_skills), embs.shape[0])
    skill2idx = {name: i for i, name in enumerate(valid_skills[:n_embs])}
    return embs, skill2idx



def derive_record(raw, skill2idx, embs, target_key, chosen_key, n_key='N', scenario=None):
    target = raw[target_key]
    chosen = raw.get(chosen_key)
    if target not in skill2idx:
        return None
    library = parse_library(raw.get('system_prompt', ''), skill2idx)
    if not library:
        return None
    comp_ids = [tid for tid in library if tid != target and tid in skill2idx]
    if not comp_ids:
        return None

    t_idx = skill2idx[target]
    t_emb = embs[t_idx]
    comp_indices = [skill2idx[tid] for tid in comp_ids]
    sims = embs[comp_indices] @ t_emb

    ordered = sorted(zip(comp_ids, sims.tolist()), key=lambda x: x[1], reverse=True)
    ordered_ids = [tid for tid, _ in ordered]
    sorted_sims = [float(sim) for _, sim in ordered]
    max_sim = sorted_sims[0]
    second_sim = sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]
    top3_mean = float(np.mean(sorted_sims[: min(3, len(sorted_sims))]))
    avg_sim = float(np.mean(sorted_sims))
    rank1_gap = float(max_sim - second_sim)
    global_ci_beta30 = float(np.sum(np.exp(30 * np.asarray(sorted_sims, dtype=float))))

    chosen_rank = None
    chosen_sim = None
    if chosen and chosen in ordered_ids:
        chosen_rank = int(ordered_ids.index(chosen) + 1)
        chosen_sim = float(sorted_sims[chosen_rank - 1])
    elif chosen == target:
        chosen_rank = 0
        chosen_sim = 1.0

    record = {
        'target': target,
        'chosen': chosen,
        'is_correct': float(raw['is_correct']),
        'N': int(raw.get(n_key, raw.get('library_size', len(library)))),
        'library_size': int(len(library)),
        'avg_sim': avg_sim,
        'max_sim': float(max_sim),
        'second_sim': float(second_sim),
        'top3_mean': top3_mean,
        'rank1_gap': rank1_gap,
        'global_ci_beta30': global_ci_beta30,
        'chosen_rank': chosen_rank,
        'chosen_sim': chosen_sim,
        'top_competitors': [
            {'skill': tid, 'sim': float(sim)} for tid, sim in ordered[:5]
        ],
    }

    for beta in BETAS:
        for k in TOPK_VALUES:
            record[f'top{k}_local_ci_beta{beta}'] = topk_local_ci(sorted_sims, beta, k)

    band_counts = {}
    for low in BAND_LOW_VALUES:
        for delta in BAND_DELTAS:
            high = min(0.95, low + delta)
            if high <= low:
                continue
            key = f'band_{low:.2f}_{high:.2f}'
            band_counts[key] = int(sum(low <= s < high for s in sorted_sims))
    record['band_counts'] = band_counts

    if scenario is not None:
        record['scenario'] = scenario

    return record



def load_f01_records(skill2idx, embs):
    records = []
    skipped = 0
    with open(F01_LOG, 'r') as f:
        for line in f:
            raw = json.loads(line)
            rec = derive_record(raw, skill2idx, embs, 'expected_skill', 'chosen_skill')
            if rec is None:
                skipped += 1
                continue
            rec['K'] = int(raw.get('K', 1))
            rec['step_idx'] = int(raw.get('step_idx', 0))
            records.append(rec)
            if ANALYSIS_RECORD_LIMIT and len(records) >= ANALYSIS_RECORD_LIMIT:
                break
    return records, skipped



def load_f05_stress_records(skill2idx, embs):
    records = []
    skipped = 0
    with open(F05_STRESS_LOG, 'r') as f:
        for line in f:
            raw = json.loads(line)
            rec = derive_record(raw, skill2idx, embs, 'target_skill', 'chosen_skill', n_key='library_size', scenario=raw.get('scenario'))
            if rec is None:
                skipped += 1
                continue
            records.append(rec)
            if ANALYSIS_RECORD_LIMIT and len(records) >= ANALYSIS_RECORD_LIMIT:
                break
    return records, skipped



def metric_summary(records, metric):
    if not records:
        return {
            'metric': metric,
            'global_spearman': {'rho': None, 'p': None},
            'global_point_biserial': {'r': None, 'p': None},
            'deciles': [],
            'threshold_tests': [],
        }
    values = [r[metric] for r in records]
    labels = [r['is_correct'] for r in records]
    summary = {
        'metric': metric,
        'global_spearman': spearman(values, labels),
        'global_point_biserial': point_biserial(values, labels),
        'deciles': decile_curve(records, metric),
    }
    thresholds = np.quantile(values, [0.2, 0.4, 0.5, 0.6, 0.8])
    tests = []
    seen = set()
    for th in thresholds:
        th = float(th)
        key = round(th, 6)
        if key in seen:
            continue
        seen.add(key)
        res = effect_size_from_threshold(values, labels, th)
        if res:
            tests.append(res)
    summary['threshold_tests'] = tests
    if tests:
        summary['best_threshold_by_gap'] = max(tests, key=lambda x: x['gap_below_minus_above'])
    return summary



def fixed_n_analysis(records, metrics):
    by_n = defaultdict(list)
    for r in records:
        by_n[r['N']].append(r)
    eligible = {N: group for N, group in by_n.items() if len(group) >= FIXED_N_MIN_COUNT}
    metric_scores = {m: [] for m in metrics}
    per_n = {}
    for N, group in sorted(eligible.items()):
        per_n[N] = {'n': len(group), 'metrics': {}}
        labels = [r['is_correct'] for r in group]
        for metric in metrics:
            vals = [r['band_counts'][metric] if metric.startswith('band_') else r[metric] for r in group]
            sp = spearman(vals, labels)
            pb = point_biserial(vals, labels)
            per_n[N]['metrics'][metric] = {
                'spearman': sp,
                'point_biserial': pb,
                'mean_metric': float(np.mean(vals)),
                'std_metric': float(np.std(vals)),
                'accuracy': float(np.mean(labels)),
            }
            if sp['rho'] is not None:
                metric_scores[metric].append(sp['rho'])
    aggregate = {}
    for metric, vals in metric_scores.items():
        if vals:
            negative_fraction = float(np.mean([v < 0 for v in vals]))
            aggregate[metric] = {
                'mean_spearman': float(np.mean(vals)),
                'median_spearman': float(np.median(vals)),
                'n_slices': int(len(vals)),
                'negative_fraction': negative_fraction,
            }
        else:
            aggregate[metric] = {
                'mean_spearman': None,
                'median_spearman': None,
                'n_slices': 0,
                'negative_fraction': None,
            }
    return per_n, aggregate



def summarise_fixed_n_ordering(named_fixed_n_per, band_fixed_n_per, metrics):
    ordering = []
    for metric in metrics:
        agg = {
            'metric': metric,
            'mean_spearman': None,
            'median_spearman': None,
            'negative_fraction': None,
            'n_slices': 0,
        }
        source = band_fixed_n_per if metric.startswith('band_') else named_fixed_n_per
        vals = []
        for entry in source.values():
            metric_info = entry['metrics'].get(metric)
            if not metric_info:
                continue
            rho = metric_info['spearman']['rho']
            if rho is not None:
                vals.append(rho)
        if vals:
            agg.update({
                'mean_spearman': float(np.mean(vals)),
                'median_spearman': float(np.median(vals)),
                'negative_fraction': float(np.mean([v < 0 for v in vals])),
                'n_slices': int(len(vals)),
            })
        ordering.append(agg)
    ordering.sort(key=lambda x: (x['mean_spearman'] is None, x['mean_spearman'] if x['mean_spearman'] is not None else math.inf))
    return ordering


def error_rank_analysis(records):
    wrong = [r for r in records if r['is_correct'] == 0.0 and r['chosen_rank'] not in (None, 0)]
    total = len(wrong)
    rank_counter = Counter(r['chosen_rank'] for r in wrong)
    top1 = sum(v for k, v in rank_counter.items() if k == 1)
    top3 = sum(v for k, v in rank_counter.items() if k <= 3)
    top5 = sum(v for k, v in rank_counter.items() if k <= 5)

    def shell_breakdown(group):
        sims = [r['chosen_sim'] for r in group if r['chosen_sim'] is not None]
        if not sims:
            return {
                'mean_chosen_sim': None,
                'shell_fraction_0.55_0.75': None,
                'shell_fraction_0.55_0.65': None,
                'shell_fraction_0.65_0.75': None,
                'shell_fraction_below_0.55': None,
                'shell_fraction_above_0.75': None,
            }
        sims_arr = np.asarray(sims, dtype=float)
        return {
            'mean_chosen_sim': float(np.mean(sims_arr)),
            'shell_fraction_0.55_0.75': float(np.mean((sims_arr >= 0.55) & (sims_arr < 0.75))),
            'shell_fraction_0.55_0.65': float(np.mean((sims_arr >= 0.55) & (sims_arr < 0.65))),
            'shell_fraction_0.65_0.75': float(np.mean((sims_arr >= 0.65) & (sims_arr < 0.75))),
            'shell_fraction_below_0.55': float(np.mean(sims_arr < 0.55)),
            'shell_fraction_above_0.75': float(np.mean(sims_arr >= 0.75)),
        }

    by_n = defaultdict(list)
    for r in wrong:
        by_n[r['N']].append(r)
    by_n_summary = {}
    for N, group in sorted(by_n.items()):
        if len(group) < 30:
            continue
        ranks = [r['chosen_rank'] for r in group]
        by_n_summary[N] = {
            'n': len(group),
            'rank1_fraction': float(np.mean([rk == 1 for rk in ranks])),
            'top3_fraction': float(np.mean([rk <= 3 for rk in ranks])),
            'top5_fraction': float(np.mean([rk <= 5 for rk in ranks])),
            'median_rank': float(np.median(ranks)),
        }
        by_n_summary[N].update(shell_breakdown(group))

    gap_vals = [r['rank1_gap'] for r in wrong]
    rank_vals = [r['chosen_rank'] for r in wrong]
    gap_rank_spearman = spearman(gap_vals, rank_vals)
    gap_rank1_pb = point_biserial(gap_vals, [1.0 if r['chosen_rank'] == 1 else 0.0 for r in wrong])

    quartiles = np.quantile(gap_vals, [0.25, 0.5, 0.75]) if gap_vals else []
    by_gap = []
    if len(quartiles) == 3:
        thresholds = [-np.inf, *quartiles, np.inf]
        labels = ['Q1 smallest gap', 'Q2', 'Q3', 'Q4 largest gap']
        for i in range(4):
            lo, hi = thresholds[i], thresholds[i + 1]
            group = [r for r in wrong if lo <= r['rank1_gap'] < hi]
            if not group:
                continue
            ranks = [r['chosen_rank'] for r in group]
            entry = {
                'bucket': labels[i],
                'low': None if not np.isfinite(lo) else float(lo),
                'high': None if not np.isfinite(hi) else float(hi),
                'n': len(group),
                'rank1_fraction': float(np.mean([rk == 1 for rk in ranks])),
                'top3_fraction': float(np.mean([rk <= 3 for rk in ranks])),
                'median_rank': float(np.median(ranks)),
            }
            entry.update(shell_breakdown(group))
            by_gap.append(entry)

    overall_shell = shell_breakdown(wrong)

    return {
        'n_wrong_with_rank': total,
        'rank_histogram': {str(k): int(v) for k, v in rank_counter.most_common(10)},
        'rank1_fraction': float(top1 / total) if total else None,
        'top3_fraction': float(top3 / total) if total else None,
        'top5_fraction': float(top5 / total) if total else None,
        'median_rank': float(np.median(rank_vals)) if rank_vals else None,
        'mean_rank': float(np.mean(rank_vals)) if rank_vals else None,
        'rank1_gap_vs_rank_spearman': gap_rank_spearman,
        'rank1_gap_vs_rank1_point_biserial': gap_rank1_pb,
        'mean_chosen_sim': overall_shell['mean_chosen_sim'],
        'shell_fraction_0.55_0.75': overall_shell['shell_fraction_0.55_0.75'],
        'shell_fraction_0.55_0.65': overall_shell['shell_fraction_0.55_0.65'],
        'shell_fraction_0.65_0.75': overall_shell['shell_fraction_0.65_0.75'],
        'shell_fraction_below_0.55': overall_shell['shell_fraction_below_0.55'],
        'shell_fraction_above_0.75': overall_shell['shell_fraction_above_0.75'],
        'by_N': by_n_summary,
        'by_rank1_gap_quartile': by_gap,
    }



def band_grid_search(records):
    labels = [r['is_correct'] for r in records]
    results = []
    band_metric_names = []
    for low in BAND_LOW_VALUES:
        for delta in BAND_DELTAS:
            high = min(0.95, low + delta)
            if high <= low:
                continue
            key = f'band_{low:.2f}_{high:.2f}'
            band_metric_names.append(key)
            vals = [r['band_counts'][key] for r in records]
            sp = spearman(vals, labels)
            pb = point_biserial(vals, labels)
            thresholds = sorted(set(vals))
            threshold_tests = []
            for th in thresholds[1:]:
                res = effect_size_from_threshold(vals, labels, th)
                if res:
                    threshold_tests.append(res)
            best_gap = max(threshold_tests, key=lambda x: x['gap_below_minus_above']) if threshold_tests else None
            results.append({
                'metric': key,
                'low': float(low),
                'high': float(high),
                'width': float(high - low),
                'spearman': sp,
                'point_biserial': pb,
                'best_threshold': best_gap,
            })
    results.sort(key=lambda x: (x['spearman']['rho'] if x['spearman']['rho'] is not None else 999))
    return results, band_metric_names



def f05_stress_validation(records, winning_metric):
    if not records:
        return {
            'winning_metric_summary': metric_summary(records, winning_metric),
            'scenario_metric_means': {},
            'scenario_accuracy_means': {},
            'old_ci_reference': None,
            'valid_for_analysis': False,
            'note': 'No F05 stress records were parseable.',
        }
    summary = metric_summary(records, winning_metric)
    scenario_groups = defaultdict(list)
    for r in records:
        scenario_groups[r['scenario']].append(r)
    scenario_metric = {}
    scenario_acc = {}
    order = ['Low', 'Mid', 'High']
    for name in order:
        group = scenario_groups.get(name, [])
        if not group:
            continue
        scenario_metric[name] = float(np.mean([r[winning_metric] for r in group]))
        scenario_acc[name] = float(np.mean([r['is_correct'] for r in group]))
    old_ci = None
    if OLD_CI_JSON.exists():
        with open(OLD_CI_JSON, 'r') as f:
            old_ci = json.load(f)
    return {
        'winning_metric_summary': summary,
        'scenario_metric_means': scenario_metric,
        'scenario_accuracy_means': scenario_acc,
        'old_ci_reference': old_ci,
    }



def compare_with_f05_reference(best_band):
    if not F05_REFERENCE_SUMMARY.exists() or not F05_REFERENCE_ULTIMATE.exists():
        return {
            'discovered_band': {'low': best_band['low'], 'high': best_band['high']},
            'available': False,
            'note': 'Optional F05 reference artifacts are not present in this open-source bundle.',
        }
    with open(F05_REFERENCE_SUMMARY, 'r') as f:
        summary = json.load(f)
    with open(F05_REFERENCE_ULTIMATE, 'r') as f:
        ultimate = json.load(f)
    return {
        'discovered_band': {'low': best_band['low'], 'high': best_band['high']},
        'F05_reference_summary': summary,
        'ultimate_combination_reference': ultimate,
        'qualitative_alignment': {
            'mid_similarity_danger_zone': 'supported' if 0.55 <= best_band['low'] <= 0.80 and 0.60 <= best_band['high'] <= 0.90 else 'unclear',
            'clone_recovery_possible': 'compatible because search excludes >=0.95 and best band can peak below clone zone',
        },
    }



def choose_winner(metric_summaries, band_results, fixed_n_agg=None):
    fixed_n_agg = fixed_n_agg or {}
    candidates = []
    for metric, summary in metric_summaries.items():
        rho = summary['global_spearman']['rho']
        fixed = summary.get('fixed_n_mean_spearman')
        if rho is None:
            continue
        candidates.append({
            'metric': metric,
            'family': 'named_metric',
            'rho': rho,
            'fixed_n_mean_spearman': fixed,
            'score': abs(rho) + 0.25 * abs(fixed if fixed is not None else 0.0),
        })
    if band_results:
        best_band = min(band_results, key=lambda x: x['spearman']['rho'] if x['spearman']['rho'] is not None else math.inf)
        band_metric = best_band['metric']
        band_fixed = fixed_n_agg.get(band_metric, {}).get('mean_spearman')
        candidates.append({
            'metric': band_metric,
            'family': 'band_count',
            'rho': best_band['spearman']['rho'],
            'fixed_n_mean_spearman': band_fixed,
            'score': abs(best_band['spearman']['rho'] or 0.0) + 0.25 * abs(band_fixed if band_fixed is not None else 0.0),
            'band': best_band,
        })
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[0], candidates



def attach_fixed_n(metric_summaries, fixed_n_agg):
    for metric, summary in metric_summaries.items():
        summary['fixed_n_mean_spearman'] = fixed_n_agg.get(metric, {}).get('mean_spearman')
        summary['fixed_n_median_spearman'] = fixed_n_agg.get(metric, {}).get('median_spearman')
        summary['fixed_n_slices'] = fixed_n_agg.get(metric, {}).get('n_slices')



def make_figure(records, metric_summaries, band_results, error_summary, f05_summary, winner):
    os.makedirs(OUT_FIG.parent, exist_ok=True)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    metric_order = ['avg_sim', 'max_sim', 'second_sim', 'top3_mean', 'top3_local_ci_beta30', 'global_ci_beta30']
    labels = []
    rhos = []
    for metric in metric_order:
        if metric in metric_summaries:
            labels.append(metric)
            rhos.append(metric_summaries[metric]['global_spearman']['rho'])
    rhos_plot = [0.0 if r is None else r for r in rhos]
    ax1.barh(range(len(labels)), rhos_plot, color=['#999999', '#d95f02', '#7570b3', '#1b9e77', '#66a61e', '#e7298a'])
    ax1.axvline(0, color='black', lw=1)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Spearman ρ vs is_correct')
    ax1.set_title('Mechanism showdown on F01')
    ax1.grid(alpha=0.25, axis='x')

    ax2 = fig.add_subplot(gs[0, 1])
    for metric, color in [('avg_sim', '#999999'), ('max_sim', '#d95f02'), ('top3_mean', '#1b9e77'), ('top3_local_ci_beta30', '#66a61e')]:
        bins = metric_summaries[metric]['deciles']
        ax2.plot([b['metric'] for b in bins], [b['acc'] * 100 for b in bins], 'o-', label=metric, color=color)
    ax2.set_xlabel('Metric value (decile mean)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy curves: avg vs decoy vs local pack')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    top_bands = band_results[:8]
    band_labels = [f"[{b['low']:.2f},{b['high']:.2f})" for b in top_bands]
    band_rhos = [b['spearman']['rho'] for b in top_bands]
    band_rhos_plot = [0.0 if r is None else r for r in band_rhos]
    ax3.barh(range(len(top_bands)), band_rhos_plot, color='#7570b3')
    ax3.set_yticks(range(len(top_bands)))
    ax3.set_yticklabels(band_labels)
    ax3.axvline(0, color='black', lw=1)
    ax3.set_xlabel('Spearman ρ vs is_correct')
    ax3.set_title('Top danger-zone bands')
    ax3.grid(alpha=0.25, axis='x')

    inset = ax3.inset_axes([0.60, 0.08, 0.36, 0.48])
    rank_hist = error_summary['rank_histogram']
    rk = [int(k) for k in rank_hist.keys()]
    rv = [rank_hist[str(k)] for k in rk]
    inset.bar(rk, rv, color='#d95f02')
    inset.set_title('Wrong-answer ranks', fontsize=10)
    inset.set_xlabel('Chosen rank', fontsize=9)
    inset.set_ylabel('Count', fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    scenario_means = f05_summary['scenario_metric_means']
    scenario_acc = f05_summary['scenario_accuracy_means']
    order = [s for s in ['Low', 'Mid', 'High'] if s in scenario_means]
    x = np.arange(len(order))
    vals = [scenario_means[s] for s in order]
    accs = [scenario_acc[s] * 100 for s in order]
    ax4.plot(x, vals, 'o-', color='#1b9e77', label=f"{winner['metric']} mean")
    ax4.set_xticks(x)
    ax4.set_xticklabels(order)
    ax4.set_ylabel('Winning metric')
    ax4.set_title('F05 validation: congestion ordering')
    ax4.grid(alpha=0.25)
    ax4b = ax4.twinx()
    ax4b.plot(x, accs, 's--', color='#d95f02', label='Accuracy (%)')
    ax4b.set_ylabel('Accuracy (%)')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

    fig.suptitle('Local Competitor Law: local neighborhood beats average crowding', fontsize=17, fontweight='bold', y=0.98)
    plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')



def main():
    print('1. Loading embeddings...')
    embs, skill2idx = load_embeddings()
    print(f'   Loaded {len(skill2idx)} mapped skills with embedding matrix {embs.shape}.')

    print('2. Parsing F01 routing log...')
    records, skipped = load_f01_records(skill2idx, embs)
    print(f'   Loaded {len(records)} records from F01 (skipped {skipped}).')

    metric_names = [
        'avg_sim', 'max_sim', 'second_sim', 'top3_mean', 'rank1_gap',
        'top2_local_ci_beta10', 'top2_local_ci_beta20', 'top2_local_ci_beta30',
        'top3_local_ci_beta10', 'top3_local_ci_beta20', 'top3_local_ci_beta30',
        'top5_local_ci_beta10', 'top5_local_ci_beta20', 'top5_local_ci_beta30',
        'global_ci_beta30',
    ]

    print('3. Computing global metric summaries...')
    metric_summaries = {metric: metric_summary(records, metric) for metric in metric_names}

    print('4. Running fixed-N analyses...')
    fixed_metrics = ['avg_sim', 'max_sim', 'second_sim', 'top3_mean', 'top3_local_ci_beta30', 'top5_local_ci_beta30', 'global_ci_beta30']
    named_fixed_n_per, named_fixed_n_agg = fixed_n_analysis(records, fixed_metrics)
    attach_fixed_n(metric_summaries, named_fixed_n_agg)

    print('5. Searching danger-zone bands...')
    band_results, band_metric_names = band_grid_search(records)
    band_fixed_n_per, band_fixed_n_agg = fixed_n_analysis(records, band_metric_names)
    best_band = band_results[0]
    best_band['fixed_n'] = {
        'aggregate': band_fixed_n_agg.get(best_band['metric'], {}),
        'per_N': {
            str(N): band_fixed_n_per[N]['metrics'][best_band['metric']]
            for N in band_fixed_n_per
            if best_band['metric'] in band_fixed_n_per[N]['metrics']
        },
    }
    fixed_n_ordering = summarise_fixed_n_ordering(
        named_fixed_n_per,
        band_fixed_n_per,
        ['avg_sim', 'max_sim', 'top3_mean', best_band['metric']],
    )
    rho = best_band.get('spearman', {}).get('rho')
    rho_text = f"{rho:.4f}" if rho is not None else "NA"
    print(f"   Best band: [{best_band['low']:.2f}, {best_band['high']:.2f}) with rho={rho_text}")

    print('6. Testing whether fatal decoy is actually fatal...')
    error_summary = error_rank_analysis(records)
    rank1 = error_summary.get('rank1_fraction')
    top3 = error_summary.get('top3_fraction')
    top5 = error_summary.get('top5_fraction')
    rank1_text = f"{rank1:.3f}" if rank1 is not None else "NA"
    top3_text = f"{top3:.3f}" if top3 is not None else "NA"
    top5_text = f"{top5:.3f}" if top5 is not None else "NA"
    print(f"   Wrong-answer rank1 fraction={rank1_text}, top3={top3_text}, top5={top5_text}")

    winner, ranking = choose_winner(metric_summaries, band_results, band_fixed_n_agg)
    print(f"7. Winning mechanism candidate: {winner['metric']} ({winner['family']})")

    print('8. Light validation on F05 stress log...')
    f05_records, f05_skipped = load_f05_stress_records(skill2idx, embs)
    f05_summary = f05_stress_validation(f05_records, winner['metric'] if winner['family'] != 'band_count' else 'top3_mean')
    print(f'   Loaded {len(f05_records)} F05 records (skipped {f05_skipped}).')

    print('9. Comparing discovered band against F05 reference...')
    f05_reference_compare = compare_with_f05_reference(best_band)

    print('10. Generating figure...')
    make_figure(records, metric_summaries, band_results, error_summary, f05_summary, winner)

    os.makedirs(OUT_JSON.parent, exist_ok=True)
    result = {
        'dataset': {
            'F01_records': len(records),
            'F01_skipped': skipped,
            'F05_records': len(f05_records),
            'F05_skipped': f05_skipped,
        },
        'metric_summaries': metric_summaries,
        'fixed_n_analysis': {
            'named_metrics': {
                'per_N': named_fixed_n_per,
                'aggregate': named_fixed_n_agg,
            },
            'band_metrics': {
                'per_N': band_fixed_n_per,
                'aggregate': band_fixed_n_agg,
            },
            'winner_band_vs_baselines': fixed_n_ordering,
        },
        'band_grid_search': band_results,
        'best_band': best_band,
        'error_rank_analysis': error_summary,
        'winner': winner,
        'winner_ranking': ranking[:10],
        'F05_validation': f05_summary,
        'F05_reference_comparison': f05_reference_compare,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'✅ Saved results: {OUT_JSON}')
    print(f'✅ Saved figure:  {OUT_FIG}')


if __name__ == '__main__':
    main()
