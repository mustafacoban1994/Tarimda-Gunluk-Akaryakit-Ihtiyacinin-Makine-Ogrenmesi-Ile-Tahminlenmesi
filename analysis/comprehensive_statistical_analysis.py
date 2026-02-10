"""
Tezdeki TÜM Confusion Matrix'leri Kullanarak Kapsamlı İstatistiksel Analiz
Doğru veriler confusion_matrices.py'dan import edilir.
"""
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confusion_matrices import (
    results, calculate_metrics, get_scenario_data, get_article_scenarios,
    get_all_scenarios, ARTICLE_SCENARIOS, SCENARIO_MAPPING,
    ALGORITHM_NAMES, VARIANT_NAMES
)

print("=" * 80)
print("TEZİN TÜM VERİLERİYLE KAPSAMLI İSTATİSTİKSEL ANALİZ")
print("(16 Senaryo x 16 Algoritma = 256 Confusion Matrix)")
print("=" * 80)

# =========================================================================
# Tüm senaryolardaki verileri topla
# =========================================================================
all_scenario_data = get_all_scenarios()
article_scenario_data = get_article_scenarios()

print("\n" + "=" * 80)
print("1. TÜM SENARYOLARDA ALGORİTMA PERFORMANSLARI")
print("=" * 80)

# Her algoritmanın tüm senaryolardaki performansını topla
algorithm_performance = {}
for scenario_name, scenario_metrics in all_scenario_data.items():
    if scenario_metrics:
        for algo, metrics in scenario_metrics.items():
            if algo not in algorithm_performance:
                algorithm_performance[algo] = []
            algorithm_performance[algo].append({
                'scenario': scenario_name,
                **metrics
            })

# Ortalama performans
print("\nAlgoritmaların Ortalama Performansı (Tüm 16 Senaryo):")
print(f"{'Algoritma':<10} {'n':>3} {'Accuracy':>12} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 65)

algo_summary = {}
for algo in sorted(algorithm_performance.keys()):
    perfs = algorithm_performance[algo]
    n = len(perfs)

    avg_acc = np.mean([p['Accuracy'] for p in perfs])
    avg_prec = np.mean([p['Precision'] for p in perfs])
    avg_rec = np.mean([p['Recall'] for p in perfs])
    avg_f1 = np.mean([p['F1'] for p in perfs])

    std_acc = np.std([p['Accuracy'] for p in perfs], ddof=1) if n > 1 else 0

    algo_summary[algo] = {
        'n': n,
        'avg_accuracy': avg_acc,
        'std_accuracy': std_acc,
        'avg_precision': avg_prec,
        'avg_recall': avg_rec,
        'avg_f1': avg_f1,
        'accuracies': [p['Accuracy'] for p in perfs]
    }

    print(f"{algo:<10} {n:3d} {avg_acc * 100:7.2f}%+/-{std_acc * 100:4.2f} "
          f"{avg_prec * 100:7.2f}% {avg_rec * 100:7.2f}% {avg_f1 * 100:7.2f}%")

# =========================================================================
# Makale senaryoları detaylı analiz (4 ana senaryo)
# =========================================================================
print("\n" + "=" * 80)
print("1b. MAKALE SENARYOLARI - DETAYLI METRİKLER")
print("=" * 80)

scenario_labels = {
    "Scenario1": "S1: Full Year, Sequential, Default",
    "Scenario2": "S2: Full Year, Sequential, Optimized",
    "Scenario3": "S3: Full Year, Percentage, Default",
    "Scenario4": "S4: Full Year, Percentage, Optimized",
}

for sname, slabel in scenario_labels.items():
    print(f"\n--- {slabel} ---")
    sdata = article_scenario_data[sname]
    df = pd.DataFrame(sdata).T
    df_display = df[['Accuracy', 'Precision', 'Recall', 'F1']].copy()
    for col in df_display.columns:
        df_display[col] = df_display[col].apply(lambda x: x * 100)
    df_display = df_display.sort_values('Accuracy', ascending=False)
    print(df_display.round(2).to_string())

print("\n" + "=" * 80)
print("2. FRIEDMAN TESTİ - ÇOKLU ALGORİTMA KARŞILAŞTIRMASI")
print("=" * 80)

# 4 makale senaryosunda ortak bulunan algoritmaları al
common_algos = None
for sname in article_scenario_data:
    sdata = article_scenario_data[sname]
    algo_set = set(sdata.keys())
    if common_algos is None:
        common_algos = algo_set
    else:
        common_algos = common_algos & algo_set

common_algos = sorted(common_algos)
print(f"\n4 senaryoda ortak {len(common_algos)} algoritma: {', '.join(common_algos)}")

# Friedman testi veri hazırlama - her algoritma için 4 senaryo accuracy
friedman_input = {}
for algo in common_algos:
    accs = []
    for sname in ["Scenario1", "Scenario2", "Scenario3", "Scenario4"]:
        sdata = article_scenario_data[sname]
        accs.append(sdata[algo]['Accuracy'] * 100)
    friedman_input[algo] = accs

print("\nAlgoritma performansları (4 senaryo):")
print(f"{'Algoritma':<10} {'S1':>8} {'S2':>8} {'S3':>8} {'S4':>8} {'Ort':>8}")
print("-" * 50)
for algo in common_algos:
    accs = friedman_input[algo]
    print(f"{algo:<10} {accs[0]:7.2f}% {accs[1]:7.2f}% {accs[2]:7.2f}% {accs[3]:7.2f}% {np.mean(accs):7.2f}%")

# Friedman testi
if len(common_algos) >= 3:
    friedman_data = [friedman_input[algo] for algo in common_algos]
    stat, p_value = friedmanchisquare(*friedman_data)

    print(f"\nFriedman Test Sonuçları:")
    print(f"  Test İstatistiği (chi2): {stat:.4f}")
    print(f"  p-değeri: {p_value:.6f}")
    print(f"  Serbestlik derecesi: {len(common_algos) - 1}")

    if p_value < 0.05:
        print(f"\n  -> SONUC: Algoritmalar arasında istatistiksel olarak ANLAMLI")
        print(f"           fark VAR (p = {p_value:.6f} < 0.05)")
    else:
        print(f"\n  -> SONUC: Algoritmalar arasında istatistiksel olarak anlamlı")
        print(f"           fark YOK (p = {p_value:.6f} >= 0.05)")

print("\n" + "=" * 80)
print("3. WILCOXON TESTİ - BASE vs FEATURE SELECTION (*) ETKİSİ")
print("=" * 80)

# Her algoritmanın base vs * versiyonunu karşılaştır
print("\nBase Algoritma vs Feature Selection (*) Karşılaştırması:")
print(f"{'Karşılaştırma':<20} {'Base Ort':>10} {'* Ort':>10} {'Fark':>10} {'p-değeri':>10}")
print("-" * 65)

wilcoxon_results = []
for base_algo in ALGORITHM_NAMES:
    star_algo = f"{base_algo}*"
    if base_algo in algo_summary and star_algo in algo_summary:
        base_accs = algo_summary[base_algo]['accuracies']
        star_accs = algo_summary[star_algo]['accuracies']

        # Match lengths
        min_len = min(len(base_accs), len(star_accs))
        if min_len >= 5:
            try:
                stat, p_val = wilcoxon(base_accs[:min_len], star_accs[:min_len])
                base_mean = np.mean(base_accs) * 100
                star_mean = np.mean(star_accs) * 100
                diff = star_mean - base_mean
                sign = "*" if p_val < 0.05 else ""
                print(f"{base_algo} vs {star_algo:<12} {base_mean:9.2f}% {star_mean:9.2f}% "
                      f"{diff:+9.2f}% {p_val:9.4f} {sign}")
                wilcoxon_results.append({
                    'comparison': f"{base_algo} vs {star_algo}",
                    'base_mean': base_mean,
                    'star_mean': star_mean,
                    'diff': diff,
                    'p_value': p_val
                })
            except Exception as e:
                print(f"{base_algo} vs {star_algo:<12} Test yapılamadı: {e}")

print("\n" + "=" * 80)
print("4. EFFECT SIZE (ETKİ BÜYÜKLÜĞÜ) ANALİZİ - Cohen's d")
print("=" * 80)


def cohens_d(group1, group2):
    """Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def interpret_d(d):
    """Cohen's d yorumu"""
    ad = abs(d)
    if ad < 0.2:
        return "Negligible"
    elif ad < 0.5:
        return "Small"
    elif ad < 0.8:
        return "Medium"
    else:
        return "Large"


# 4 makale senaryosu üzerinden effect size
print("\nMakale Senaryolarında Algoritma Çiftleri İçin Effect Size:")
print(f"{'Karşılaştırma':<25} {'Cohen d':>12} {'Yorum':>12}")
print("-" * 55)

# En iyi algoritmayı bul (4 senaryo ortalaması)
algo_avg_4s = {}
for algo in common_algos:
    algo_avg_4s[algo] = np.mean(friedman_input[algo])

sorted_algos = sorted(algo_avg_4s.items(), key=lambda x: x[1], reverse=True)
best_algo = sorted_algos[0][0]

print(f"\nEn iyi algoritma (4 senaryo): {best_algo} ({algo_avg_4s[best_algo]:.2f}%)")
print(f"\n{best_algo} ile diğerleri:")

effect_sizes = []
for algo, avg in sorted_algos[1:]:
    d = cohens_d(friedman_input[best_algo], friedman_input[algo])
    interp = interpret_d(d)
    effect_sizes.append({
        'comparison': f"{best_algo} vs {algo}",
        'cohens_d': d,
        'interpretation': interp
    })
    print(f"  {best_algo} vs {algo:<15} {d:12.4f} {interp:>12}")

print("\n" + "=" * 80)
print("5. GÜVEN ARALIKLARI (CONFIDENCE INTERVALS)")
print("=" * 80)

print("\n4 Makale Senaryosu Üzerinden %95 Güven Aralıkları:")
print(f"{'Algoritma':<10} {'Ortalama':>10} {'%95 CI':>25}")
print("-" * 50)

ci_results = {}
for algo in common_algos:
    accs = friedman_input[algo]
    mean_val = np.mean(accs)
    if len(accs) >= 2:
        sem = stats.sem(accs)
        ci = stats.t.interval(0.95, len(accs) - 1, loc=mean_val, scale=sem)
        ci_results[algo] = {'mean': mean_val, 'ci_low': ci[0], 'ci_high': ci[1]}
        print(f"{algo:<10} {mean_val:9.2f}% [{ci[0]:6.2f}%, {ci[1]:6.2f}%]")
    else:
        print(f"{algo:<10} {mean_val:9.2f}% [Tek veri noktası]")

print("\n" + "=" * 80)
print("6. PERFORMANS STABİLİTESİ ANALİZİ")
print("=" * 80)

print("\nAlgoritmaların Farklı Senaryolardaki Stabilitesi:")
print("(Düşük CV = Daha stabil performans)")
print(f"\n{'Algoritma':<10} {'Ort. Accuracy':>15} {'Std':>10} {'CV':>8} {'Stabilite':>14}")
print("-" * 65)

stability_data = []
for algo in common_algos:
    accs = friedman_input[algo]
    mean_val = np.mean(accs)
    std_val = np.std(accs, ddof=1) if len(accs) > 1 else 0
    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0

    if cv < 5:
        stability = "Cok Stabil"
    elif cv < 10:
        stability = "Stabil"
    elif cv < 20:
        stability = "Orta"
    else:
        stability = "Degisken"

    stability_data.append({
        'algo': algo, 'mean': mean_val, 'std': std_val,
        'cv': cv, 'stability': stability
    })
    print(f"{algo:<10} {mean_val:12.2f}% {std_val:9.2f}% {cv:7.2f}% {stability:>14}")

print("\n" + "=" * 80)
print("7. KAPSAMLI KARŞILAŞTIRMA TABLOSU (TÜM 16 SENARYO)")
print("=" * 80)

# Tüm algoritmalar, 16 senaryo ortalaması
print("\nTüm Algoritmalar - 16 Senaryo Ortalaması:")
print(f"{'Algoritma':<10} {'n':>3} {'Accuracy':>12} {'Precision':>12} {'Recall':>10} {'F1':>10} {'MCC':>8}")
print("-" * 70)

sorted_all = sorted(algo_summary.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)
for algo, summary in sorted_all:
    print(f"{algo:<10} {summary['n']:3d} {summary['avg_accuracy'] * 100:7.2f}%+/-{summary['std_accuracy'] * 100:4.2f} "
          f"{summary['avg_precision'] * 100:7.2f}% {summary['avg_recall'] * 100:7.2f}% "
          f"{summary['avg_f1'] * 100:7.2f}%")

print("\n" + "=" * 80)
print("SONUÇ VE DEĞERLENDİRME")
print("=" * 80)

print(f"""
Kapsamli Istatistiksel Analiz Tamamlandi!
Toplam: {len(all_scenario_data)} senaryo x 16 algoritma = {len(all_scenario_data) * 16} confusion matrix

Yapilan Analizler:
  1. Tum senaryolarda algoritma performanslarinin hesaplanmasi
  2. Friedman testi ile coklu algoritma karsilastirmasi
  3. Wilcoxon testi ile base vs feature selection (*) etkisi
  4. Effect size (Cohen's d) analizi
  5. %95 guven araliklarinin hesaplanmasi
  6. Performans stabilitesi degerlendirmesi
  7. Kapsamli metrik karsilastirmalari

Veri Kaynagi: Mustafa COBAN YL Tezi, KTO Karatay Universitesi, 2021
              EK 1: Hata Matrisleri (Tablo 31-46, Sayfa 122-137)
""")
print("=" * 80)
