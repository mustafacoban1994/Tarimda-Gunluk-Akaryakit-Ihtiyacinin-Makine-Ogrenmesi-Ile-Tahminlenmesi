#!/usr/bin/env python3
"""
Generates publication-quality figures for the article.
All metrics are computed dynamically from confusion_matrices.py.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confusion_matrices import get_article_scenarios, ALGORITHM_NAMES

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

output_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
article_scenarios = get_article_scenarios()
scenario_names = ["Scenario1", "Scenario2", "Scenario3", "Scenario4"]
scenario_labels = ["S1:\nSequential\nDefault", "S2:\nSequential\nOptimized",
                    "S3:\nPercentage\nDefault", "S4:\nPercentage\nOptimized"]

common_algos = None
for sname in scenario_names:
    algo_set = set(a for a in article_scenarios[sname] if not a.endswith('*'))
    common_algos = algo_set if common_algos is None else common_algos & algo_set
common_algos = sorted(common_algos)

algo_scenario_accs = {}
for algo in common_algos:
    algo_scenario_accs[algo] = [
        article_scenarios[sname][algo]['Accuracy'] * 100 for sname in scenario_names
    ]

algo_stats = {}
for algo in common_algos:
    accs = algo_scenario_accs[algo]
    algo_stats[algo] = {
        'mean': np.mean(accs),
        'std': np.std(accs, ddof=1) if len(accs) > 1 else 0,
    }

sorted_algos = sorted(algo_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
algorithms = [a[0] for a in sorted_algos]
mean_accuracy = [a[1]['mean'] for a in sorted_algos]
std_dev = [a[1]['std'] for a in sorted_algos]
cv_values = [(s / m) * 100 if m > 0 else 0 for m, s in zip(mean_accuracy, std_dev)]

# ---------------------------------------------------------------------------
# Figure 1: Algorithm Performance Comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(algorithms))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(algorithms)))

bars = ax.bar(x_pos, mean_accuracy, yerr=std_dev,
              color=colors, alpha=0.8, capsize=5,
              edgecolor='black', linewidth=1.5)

for i in range(min(3, len(bars))):
    bars[i].set_edgecolor('red')
    bars[i].set_linewidth(2.5)

ax.set_xlabel('Algorithms', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Algorithm Performance Comparison with 95% Confidence Intervals',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, fontsize=10, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

for i, (mean, std) in enumerate(zip(mean_accuracy, std_dev)):
    ax.text(i, mean + std + 2, f'{mean:.1f}%\n+/-{std:.1f}', ha='center',
            va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure1_algorithm_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Performance Across Scenarios
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
top5 = algorithms[:5]
markers = ['o', 's', '^', 'D', 'v']
colors_line = ['#2E86AB', '#A23B72', '#F18F01', '#2CA58D', '#E63946']
x_pos = np.arange(len(scenario_labels))

for i, algo in enumerate(top5):
    ax.plot(x_pos, algo_scenario_accs[algo], marker=markers[i], linewidth=2.5,
            markersize=10, label=algo, color=colors_line[i])

ax.set_xlabel('Experimental Scenarios', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Top 5 Algorithm Performance Across Different Scenarios',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_labels, fontsize=11)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

all_accs = [a for algo in top5 for a in algo_scenario_accs[algo]]
ax.set_ylim(max(0, min(all_accs) - 10), min(100, max(all_accs) + 10))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure2_scenario_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Stability Analysis (CV vs Accuracy)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
sizes = [acc * 8 for acc in mean_accuracy]

scatter = ax.scatter(cv_values, mean_accuracy, s=sizes, alpha=0.6,
                     c=mean_accuracy, cmap='RdYlGn', edgecolors='black', linewidth=2)

for i, (cv, acc, algo) in enumerate(zip(cv_values, mean_accuracy, algorithms)):
    ax.annotate(algo, (cv, acc), fontsize=10, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')

ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60% Accuracy Threshold')
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='10% CV Threshold')
ax.set_xlabel('Coefficient of Variation (CV) %', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Algorithm Stability vs Performance Trade-off',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Mean Accuracy (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure3_stability_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Effect Size Visualization
# ---------------------------------------------------------------------------
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0


best_algo = algorithms[0]
comparisons_list, d_values, interps = [], [], []

for algo in algorithms[1:]:
    d = cohens_d(algo_scenario_accs[best_algo], algo_scenario_accs[algo])
    comparisons_list.append(f"{best_algo} vs {algo}")
    d_values.append(d)
    ad = abs(d)
    interps.append("Negligible" if ad < 0.2 else "Small" if ad < 0.5
                    else "Medium" if ad < 0.8 else "Large")

fig, ax = plt.subplots(figsize=(10, max(7, len(comparisons_list) * 0.6)))
colors_effect = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(comparisons_list)))
y_pos = np.arange(len(comparisons_list))

ax.barh(y_pos, d_values, color=colors_effect, alpha=0.7,
        edgecolor='black', linewidth=1.5)

ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Small (0.2)')
ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Medium (0.5)')
ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Large (0.8)')

ax.set_yticks(y_pos)
ax.set_yticklabels(comparisons_list, fontsize=10)
ax.set_xlabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
ax.set_title("Effect Size Analysis: Magnitude of Performance Differences",
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, (d, interp) in enumerate(zip(d_values, interps)):
    ax.text(d + 0.05, i, f'd = {d:.2f} ({interp})',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure4_effect_size.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Data Encoding Impact (Sequential vs Percentage)
# ---------------------------------------------------------------------------
s1_data = article_scenarios["Scenario1"]
s3_data = article_scenarios["Scenario3"]

enc_algos = sorted(a for a in set(s1_data) & set(s3_data) if not a.endswith('*'))
enc_diffs = [(a, s3_data[a]['Accuracy'] * 100 - s1_data[a]['Accuracy'] * 100) for a in enc_algos]
enc_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
top_enc = [a[0] for a in enc_diffs[:8]]

seq_perf = [s1_data[algo]['Accuracy'] * 100 for algo in top_enc]
pct_perf = [s3_data[algo]['Accuracy'] * 100 for algo in top_enc]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(top_enc))
width = 0.35

ax.bar(x - width / 2, seq_perf, width, label='Sequential Encoding',
       color='#87CEEB', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.bar(x + width / 2, pct_perf, width, label='Percentage Encoding',
       color='#32CD32', alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (s_p, p_p) in enumerate(zip(seq_perf, pct_perf)):
    change = p_p - s_p
    color = 'green' if change > 0 else 'red'
    sign = '+' if change > 0 else ''
    ax.text(i, max(s_p, p_p) + 2, f'{sign}{change:.1f}%',
            ha='center', fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('Algorithms', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Impact of Data Encoding on Performance (Sequential vs Percentage)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(top_enc, fontsize=10, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure5_optimization_impact.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Figure 6: Box Plot Distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))
data_list = [algo_scenario_accs[algo] for algo in algorithms]

bp = ax.boxplot(data_list, labels=algorithms, patch_artist=True,
                notch=False, showmeans=True, meanline=True)

colors_box = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Algorithms', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Algorithm Performance Across 4 Scenarios',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')

all_box_data = [v for d in data_list for v in d]
ax.set_ylim(max(0, min(all_box_data) - 10), min(100, max(all_box_data) + 10))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure6_boxplot_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"All 6 figures saved to {output_dir}")
