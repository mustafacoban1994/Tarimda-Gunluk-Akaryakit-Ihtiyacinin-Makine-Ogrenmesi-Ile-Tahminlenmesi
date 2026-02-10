#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for Article
Computes all metrics from confusion matrix data for Table 22-25 (4 article scenarios).
"""

import sys
import os
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confusion_matrices import results, calculate_metrics, get_scenario_data, get_article_scenarios, ARTICLE_SCENARIOS, ALGORITHM_NAMES

# =========================================================================
# 1. Compute per-scenario metrics for all 16 algorithm variants
# =========================================================================

scenarios = get_article_scenarios()
scenario_names = list(ARTICLE_SCENARIOS.keys())
table_nums = list(ARTICLE_SCENARIOS.values())

# Build full variant labels: 8 base + 8 optimized = 16
algo_labels = []
for algo in ALGORITHM_NAMES:
    algo_labels.append(algo)        # normal variant
    algo_labels.append(f"{algo}*")  # optimized variant (feature selection)

print("=" * 100)
print("COMPREHENSIVE ARTICLE STATISTICS FROM CONFUSION MATRICES")
print("=" * 100)

# =========================================================================
# Section 1: Per-scenario metrics for each of 16 algorithm variants
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 1: PER-SCENARIO METRICS (Accuracy, Precision, Recall, F1)")
print("=" * 100)

# Store accuracies for later analysis
# accuracy_matrix[algo_label] = [acc_s1, acc_s2, acc_s3, acc_s4]
accuracy_matrix = {}
precision_matrix = {}
recall_matrix = {}
f1_matrix = {}

for label in algo_labels:
    accuracy_matrix[label] = []
    precision_matrix[label] = []
    recall_matrix[label] = []
    f1_matrix[label] = []

for sname, day in ARTICLE_SCENARIOS.items():
    print(f"\n--- {sname} (Table {day}) ---")
    print(f"{'Algorithm':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}  |  {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
    print("-" * 90)

    scenario_data = get_scenario_data(day)
    for label in algo_labels:
        if label in scenario_data:
            m = scenario_data[label]
            accuracy_matrix[label].append(m['Accuracy'])
            precision_matrix[label].append(m['Precision'])
            recall_matrix[label].append(m['Recall'])
            f1_matrix[label].append(m['F1'])
            print(f"{label:<12} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f}  |  {m['TN']:>5} {m['FP']:>5} {m['FN']:>5} {m['TP']:>5}")
        else:
            print(f"{label:<12} {'N/A':>10}")

# =========================================================================
# Section 2: Average metrics across 4 scenarios
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 2: AVERAGE METRICS ACROSS 4 SCENARIOS")
print("=" * 100)

avg_accuracy = {}
avg_precision = {}
avg_recall = {}
avg_f1 = {}

print(f"\n{'Algorithm':<12} {'Avg Acc':>10} {'Avg Prec':>10} {'Avg Rec':>10} {'Avg F1':>10} {'Std Acc':>10}")
print("-" * 72)

for label in algo_labels:
    if len(accuracy_matrix[label]) == 4:
        avg_accuracy[label] = np.mean(accuracy_matrix[label])
        avg_precision[label] = np.mean(precision_matrix[label])
        avg_recall[label] = np.mean(recall_matrix[label])
        avg_f1[label] = np.mean(f1_matrix[label])
        std_acc = np.std(accuracy_matrix[label], ddof=1)
        print(f"{label:<12} {avg_accuracy[label]:>10.4f} {avg_precision[label]:>10.4f} {avg_recall[label]:>10.4f} {avg_f1[label]:>10.4f} {std_acc:>10.4f}")

# =========================================================================
# Section 3: Ranking by average accuracy
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 3: RANKING BY AVERAGE ACCURACY (Best to Worst)")
print("=" * 100)

sorted_algos = sorted(avg_accuracy.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Algorithm':<12} {'Avg Accuracy':>14} {'Avg Precision':>14} {'Avg Recall':>12} {'Avg F1':>10}")
print("-" * 72)
for rank, (label, avg_acc) in enumerate(sorted_algos, 1):
    print(f"{rank:<6} {label:<12} {avg_acc:>14.4f} {avg_precision[label]:>14.4f} {avg_recall[label]:>12.4f} {avg_f1[label]:>10.4f}")

# =========================================================================
# Section 4: Best algorithm and Top 5
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 4: BEST ALGORITHM & TOP 5")
print("=" * 100)

best_label, best_avg = sorted_algos[0]
print(f"\nBest Algorithm: {best_label}")
print(f"Best Average Accuracy: {best_avg:.4f} ({best_avg*100:.2f}%)")

print(f"\nTop 5 Algorithms:")
for rank, (label, avg_acc) in enumerate(sorted_algos[:5], 1):
    accs_str = ", ".join([f"{a:.4f}" for a in accuracy_matrix[label]])
    print(f"  {rank}. {label:<12} Avg={avg_acc:.4f} ({avg_acc*100:.2f}%)  Scenarios: [{accs_str}]")

# =========================================================================
# Section 5: Friedman Test
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 5: FRIEDMAN TEST")
print("=" * 100)

# Friedman test expects groups of related samples
# Each row = scenario, each column = algorithm
# We need arrays where each array is accuracies of one algorithm across scenarios
friedman_data = []
friedman_labels = []
for label in algo_labels:
    if len(accuracy_matrix[label]) == 4:
        friedman_data.append(accuracy_matrix[label])
        friedman_labels.append(label)

# scipy.stats.friedmanchisquare expects each group as a separate argument
stat, p_value = stats.friedmanchisquare(*friedman_data)
df = len(friedman_labels) - 1  # k - 1

print(f"\nFriedman Test (k={len(friedman_labels)} algorithms, n={4} scenarios):")
print(f"  Chi-squared statistic: {stat:.4f}")
print(f"  Degrees of freedom (df): {df}")
print(f"  p-value: {p_value:.6f}")
print(f"  p-value (scientific): {p_value:.4e}")
if p_value < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05) - reject null hypothesis")
    print(f"  Interpretation: There are statistically significant differences among algorithms.")
else:
    print(f"  Result: NOT SIGNIFICANT (p >= 0.05) - fail to reject null hypothesis")
    print(f"  Interpretation: No statistically significant differences among algorithms.")

# =========================================================================
# Section 6: Cohen's d (best vs each of top algorithms)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 6: COHEN'S d EFFECT SIZE (Best vs. Others in Top 5)")
print("=" * 100)

def cohens_d(group1, group2):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def interpret_cohens_d(d):
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

best_accs = accuracy_matrix[best_label]

print(f"\nReference (best): {best_label} accuracies = {[f'{a:.4f}' for a in best_accs]}")
print(f"\n{'Comparison':<30} {'Cohen d':>10} {'Effect Size':>15} {'Best Mean':>10} {'Other Mean':>12}")
print("-" * 80)

for label, avg_acc in sorted_algos[1:]:  # all others
    other_accs = accuracy_matrix[label]
    d = cohens_d(best_accs, other_accs)
    interp = interpret_cohens_d(d)
    print(f"{best_label + ' vs ' + label:<30} {d:>10.4f} {interp:>15} {np.mean(best_accs):>10.4f} {np.mean(other_accs):>12.4f}")

# Also show specifically top 5
print(f"\n--- Cohen's d: Best vs Top 5 only ---")
for rank, (label, avg_acc) in enumerate(sorted_algos[1:5], 2):
    other_accs = accuracy_matrix[label]
    d = cohens_d(best_accs, other_accs)
    interp = interpret_cohens_d(d)
    print(f"  Rank {rank}: {best_label} vs {label}: d = {d:.4f} ({interp})")

# =========================================================================
# Section 7: 95% Confidence Interval for best algorithm
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 7: 95% CONFIDENCE INTERVAL FOR BEST ALGORITHM")
print("=" * 100)

best_arr = np.array(best_accs)
n = len(best_arr)
mean = np.mean(best_arr)
std_err = stats.sem(best_arr)
std_dev = np.std(best_arr, ddof=1)

# t-distribution CI
t_crit = stats.t.ppf(0.975, df=n-1)
ci_lower = mean - t_crit * std_err
ci_upper = mean + t_crit * std_err

print(f"\nAlgorithm: {best_label}")
print(f"  Accuracies: {[f'{a:.4f}' for a in best_accs]}")
print(f"  Mean: {mean:.4f} ({mean*100:.2f}%)")
print(f"  Std Dev: {std_dev:.4f}")
print(f"  Std Error: {std_err:.4f}")
print(f"  t-critical (df={n-1}, 95%): {t_crit:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# Also compute CI for all algorithms in top 5
print(f"\n--- 95% CI for Top 5 Algorithms ---")
for rank, (label, avg_acc) in enumerate(sorted_algos[:5], 1):
    arr = np.array(accuracy_matrix[label])
    m = np.mean(arr)
    se = stats.sem(arr)
    sd = np.std(arr, ddof=1)
    tc = stats.t.ppf(0.975, df=len(arr)-1)
    lo = m - tc * se
    hi = m + tc * se
    print(f"  {rank}. {label:<12}: Mean={m:.4f}, Std={sd:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

# =========================================================================
# Section 8: Wilcoxon Signed-Rank Test (best vs second best)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 8: WILCOXON SIGNED-RANK TEST (Best vs Second Best)")
print("=" * 100)

second_label, second_avg = sorted_algos[1]
best_arr_w = np.array(accuracy_matrix[best_label])
second_arr_w = np.array(accuracy_matrix[second_label])

print(f"\nComparing: {best_label} vs {second_label}")
print(f"  {best_label} accuracies:  {[f'{a:.4f}' for a in best_arr_w]}")
print(f"  {second_label} accuracies: {[f'{a:.4f}' for a in second_arr_w]}")
print(f"  Differences: {[f'{a-b:.4f}' for a, b in zip(best_arr_w, second_arr_w)]}")

try:
    # Wilcoxon signed-rank test
    w_stat, w_pvalue = stats.wilcoxon(best_arr_w, second_arr_w)
    print(f"\n  Wilcoxon W statistic: {w_stat:.4f}")
    print(f"  p-value: {w_pvalue:.6f}")
    print(f"  p-value (scientific): {w_pvalue:.4e}")
    if w_pvalue < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")
except Exception as e:
    print(f"  Wilcoxon test error: {e}")
    print(f"  Note: With only 4 samples, Wilcoxon may not have sufficient data.")

# Also try best vs 3rd, 4th, 5th
print(f"\n--- Wilcoxon: Best vs other top algorithms ---")
for rank, (label, avg_acc) in enumerate(sorted_algos[1:5], 2):
    other_arr = np.array(accuracy_matrix[label])
    try:
        w_stat, w_pvalue = stats.wilcoxon(best_arr_w, other_arr)
        sig = "SIGNIFICANT" if w_pvalue < 0.05 else "NOT significant"
        print(f"  {best_label} vs {label}: W={w_stat:.4f}, p={w_pvalue:.6f} ({sig})")
    except Exception as e:
        print(f"  {best_label} vs {label}: Error - {e}")

# =========================================================================
# Section 9: Feature Selection Impact (normal vs optimized)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 9: FEATURE SELECTION IMPACT (Normal vs Optimized/*)")
print("=" * 100)

print(f"\n{'Algorithm':<10} {'Normal Avg':>12} {'Optim Avg':>12} {'Difference':>12} {'% Change':>10} {'Better':>10}")
print("-" * 70)

for algo in ALGORITHM_NAMES:
    normal_label = algo
    optim_label = f"{algo}*"
    if normal_label in avg_accuracy and optim_label in avg_accuracy:
        norm_avg = avg_accuracy[normal_label]
        opt_avg = avg_accuracy[optim_label]
        diff = opt_avg - norm_avg
        pct_change = (diff / norm_avg) * 100 if norm_avg != 0 else 0
        better = "Optimized" if diff > 0 else ("Normal" if diff < 0 else "Equal")
        print(f"{algo:<10} {norm_avg:>12.4f} {opt_avg:>12.4f} {diff:>12.4f} {pct_change:>9.2f}% {better:>10}")

# Detailed per-scenario
print(f"\n--- Per-Scenario Feature Selection Impact ---")
for algo in ALGORITHM_NAMES:
    normal_label = algo
    optim_label = f"{algo}*"
    print(f"\n  {algo}:")
    for i, (sname, day) in enumerate(ARTICLE_SCENARIOS.items()):
        n_acc = accuracy_matrix[normal_label][i]
        o_acc = accuracy_matrix[optim_label][i]
        diff = o_acc - n_acc
        print(f"    {sname} (Table {day}): Normal={n_acc:.4f}, Optimized={o_acc:.4f}, Diff={diff:+.4f}")

# =========================================================================
# Section 10: Stability Analysis (Std Dev across scenarios)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 10: STABILITY ANALYSIS (Std Dev of Accuracy Across 4 Scenarios)")
print("=" * 100)

print(f"\n{'Algorithm':<12} {'Mean Acc':>10} {'Std Dev':>10} {'Min Acc':>10} {'Max Acc':>10} {'Range':>10} {'CV%':>8}")
print("-" * 72)

stability_data = []
for label in algo_labels:
    if len(accuracy_matrix[label]) == 4:
        arr = np.array(accuracy_matrix[label])
        mean_a = np.mean(arr)
        std_a = np.std(arr, ddof=1)
        min_a = np.min(arr)
        max_a = np.max(arr)
        range_a = max_a - min_a
        cv = (std_a / mean_a * 100) if mean_a != 0 else 0
        stability_data.append((label, mean_a, std_a, min_a, max_a, range_a, cv))
        print(f"{label:<12} {mean_a:>10.4f} {std_a:>10.4f} {min_a:>10.4f} {max_a:>10.4f} {range_a:>10.4f} {cv:>7.2f}%")

# Rank by stability (lowest std dev = most stable)
print(f"\n--- Algorithms Ranked by Stability (lowest Std Dev) ---")
stability_sorted = sorted(stability_data, key=lambda x: x[2])
for rank, (label, mean_a, std_a, min_a, max_a, range_a, cv) in enumerate(stability_sorted, 1):
    print(f"  {rank}. {label:<12} Std={std_a:.4f}, Mean={mean_a:.4f}, CV={cv:.2f}%")

# =========================================================================
# Section 11: Summary table for article (all metrics in compact form)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 11: COMPACT SUMMARY TABLE FOR ARTICLE")
print("=" * 100)

print(f"\n{'Algo':<10}", end="")
for sname in scenario_names:
    print(f" | {sname[:4]} Acc", end="")
print(f" | {'Avg Acc':>8} | {'Rank':>4}")
print("-" * 90)

for rank, (label, avg_acc) in enumerate(sorted_algos, 1):
    print(f"{label:<10}", end="")
    for i in range(4):
        print(f" | {accuracy_matrix[label][i]:>8.4f}", end="")
    print(f" | {avg_acc:>8.4f} | {rank:>4}")

# =========================================================================
# Section 12: Percentage-formatted values for direct article use
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 12: PERCENTAGE VALUES FOR DIRECT ARTICLE USE")
print("=" * 100)

print(f"\n{'Algo':<10}", end="")
for sname in scenario_names:
    print(f" | {sname[:4]} Acc%", end="")
print(f" | {'Avg Acc%':>8} | {'Rank':>4}")
print("-" * 90)

for rank, (label, avg_acc) in enumerate(sorted_algos, 1):
    print(f"{label:<10}", end="")
    for i in range(4):
        print(f" | {accuracy_matrix[label][i]*100:>8.2f}", end="")
    print(f" | {avg_acc*100:>8.2f} | {rank:>4}")

# =========================================================================
# Section 13: Additional pairwise statistical tests
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 13: NEMENYI POST-HOC TEST (Rank-based)")
print("=" * 100)

# Compute ranks per scenario
print("\n--- Ranks per Scenario ---")
rank_matrix = {}
for label in algo_labels:
    rank_matrix[label] = []

for i, (sname, day) in enumerate(ARTICLE_SCENARIOS.items()):
    # Get accuracies for this scenario
    accs = [(label, accuracy_matrix[label][i]) for label in algo_labels]
    # Rank: highest accuracy = rank 1
    accs_sorted = sorted(accs, key=lambda x: x[1], reverse=True)
    ranks = {}
    for r, (lbl, acc) in enumerate(accs_sorted, 1):
        ranks[lbl] = r

    print(f"\n  {sname} (Table {day}):")
    for lbl, acc in accs_sorted:
        rank_matrix[lbl].append(ranks[lbl])
        print(f"    Rank {ranks[lbl]:>2}: {lbl:<12} Acc={acc:.4f}")

print(f"\n--- Average Ranks ---")
avg_ranks = {}
for label in algo_labels:
    if len(rank_matrix[label]) == 4:
        avg_ranks[label] = np.mean(rank_matrix[label])

avg_ranks_sorted = sorted(avg_ranks.items(), key=lambda x: x[1])
for label, avg_r in avg_ranks_sorted:
    ranks_str = ", ".join([str(r) for r in rank_matrix[label]])
    print(f"  {label:<12} Avg Rank = {avg_r:.2f}  Ranks: [{ranks_str}]")

# =========================================================================
# Section 14: Mann-Whitney U tests (supplementary)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 14: PAIRED T-TEST (Best vs Others)")
print("=" * 100)

for rank, (label, avg_acc) in enumerate(sorted_algos[1:5], 2):
    other_arr = np.array(accuracy_matrix[label])
    t_stat, t_pval = stats.ttest_rel(best_arr_w, other_arr)
    print(f"  {best_label} vs {label}: t={t_stat:.4f}, p={t_pval:.6f} {'*' if t_pval < 0.05 else ''}")

# =========================================================================
# Section 15: Full Detailed Metrics (Precision, Recall, F1 per scenario)
# =========================================================================

print("\n" + "=" * 100)
print("SECTION 15: FULL METRICS TABLE (All Scenarios, All Algorithms)")
print("=" * 100)

for sname, day in ARTICLE_SCENARIOS.items():
    print(f"\n--- {sname} (Table {day}) ---")
    print(f"{'Algo':<12} {'Accuracy%':>10} {'Precision%':>11} {'Recall%':>9} {'F1%':>8} {'Specificity%':>13} {'MCC':>8}")
    print("-" * 75)
    scenario_data = get_scenario_data(day)
    for label in algo_labels:
        if label in scenario_data:
            m = scenario_data[label]
            print(f"{label:<12} {m['Accuracy']*100:>10.2f} {m['Precision']*100:>11.2f} {m['Recall']*100:>9.2f} {m['F1']*100:>8.2f} {m['Specificity']*100:>13.2f} {m['MCC']:>8.4f}")

# =========================================================================
# Final summary
# =========================================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY OF KEY FINDINGS")
print("=" * 100)

print(f"""
1. BEST ALGORITHM: {best_label}
   Average Accuracy: {best_avg:.4f} ({best_avg*100:.2f}%)

2. TOP 5 ALGORITHMS:""")
for rank, (label, avg_acc) in enumerate(sorted_algos[:5], 1):
    print(f"   {rank}. {label:<12} {avg_acc:.4f} ({avg_acc*100:.2f}%)")

print(f"""
3. FRIEDMAN TEST:
   Chi2 = {stat:.4f}, df = {df}, p = {p_value:.6f}
   {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05

4. COHEN'S d (Best vs Top 5):""")
for rank, (label, avg_acc) in enumerate(sorted_algos[1:5], 2):
    d = cohens_d(best_accs, accuracy_matrix[label])
    print(f"   {best_label} vs {label}: d = {d:.4f} ({interpret_cohens_d(d)})")

print(f"""
5. 95% CONFIDENCE INTERVAL ({best_label}):
   [{ci_lower:.4f}, {ci_upper:.4f}] = [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]

6. WILCOXON ({best_label} vs {second_label}):""")
try:
    w_stat2, w_pvalue2 = stats.wilcoxon(best_arr_w, second_arr_w)
    print(f"   W = {w_stat2:.4f}, p = {w_pvalue2:.6f}")
except Exception as e:
    print(f"   Error: {e}")

print(f"""
7. FEATURE SELECTION IMPACT (Normal vs Optimized avg accuracy):""")
for algo in ALGORITHM_NAMES:
    norm_avg = avg_accuracy[algo]
    opt_avg = avg_accuracy[f"{algo}*"]
    diff = opt_avg - norm_avg
    pct = (diff / norm_avg * 100) if norm_avg != 0 else 0
    arrow = "+" if diff > 0 else ""
    print(f"   {algo:<6}: Normal={norm_avg:.4f}, Optimized={opt_avg:.4f}, Diff={arrow}{diff:.4f} ({arrow}{pct:.2f}%)")

print(f"""
8. STABILITY (Top 5 most stable by Std Dev):""")
for rank, (label, mean_a, std_a, min_a, max_a, range_a, cv) in enumerate(stability_sorted[:5], 1):
    print(f"   {rank}. {label:<12} Std={std_a:.4f}, CV={cv:.2f}%")

print("\n" + "=" * 100)
print("END OF ANALYSIS")
print("=" * 100)
