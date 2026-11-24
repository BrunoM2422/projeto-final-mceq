#!/usr/bin/env python3
"""Reproduz a análise da base METR para o Projeto Final MCEQ.

Desfecho primário planejado no Marco 2:
  initial_implementation_time ~ ai_treatment + predicted_time_no_ai

Desfecho secundário (reprodução do resultado publicado pela METR):
  total_implementation_time = initial + post-review, com imputação da média
  pós-revisão por grupo, conforme regression.py da METR.

O script baixa data_complete.csv do repositório original se o arquivo não existir.
Não altera a fonte bruta; arquivos derivados são gravados em resultados/.
"""
from pathlib import Path
from urllib.request import urlretrieve
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs/main/data_complete.csv"
DATA_FILE = Path("data/brutos/data_complete.csv")
OUT = Path("resultados")
ALPHA = 0.05


def load_data():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print(f"Baixando dados de: {DATA_URL}")
        urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    required = ["dev_id", "issue_id", "predicted_time_no_ai", "ai_treatment",
                "initial_implementation_time", "post_review_implementation_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}. Use data_complete.csv da METR.")
    df["ai_treatment"] = pd.to_numeric(df["ai_treatment"], errors="coerce")
    for c in ["predicted_time_no_ai", "initial_implementation_time", "post_review_implementation_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dev_id", "issue_id", "ai_treatment", "predicted_time_no_ai", "initial_implementation_time"]).copy()
    df = df[(df["predicted_time_no_ai"] > 0) & (df["initial_implementation_time"] > 0)].copy()
    return df


def fit_and_report(data, outcome, label):
    d = data.dropna(subset=[outcome, "predicted_time_no_ai", "ai_treatment", "dev_id"]).copy()
    d = d[(d[outcome] > 0) & (d["predicted_time_no_ai"] > 0)].copy()
    formula = f"np.log({outcome}) ~ ai_treatment + np.log(predicted_time_no_ai)"
    model = smf.ols(formula, data=d, missing="drop")
    specs = {
        "HC3": model.fit(cov_type="HC3"),
        "Cluster_dev_id": model.fit(cov_type="cluster", cov_kwds={"groups": d["dev_id"]}),
        "Nao_robusto": model.fit(),
    }
    rows = []
    print(f"\n=== {label} ===")
    print(f"N = {len(d)} tarefas; desenvolvedores = {d['dev_id'].nunique()}")
    for name, res in specs.items():
        beta = float(res.params["ai_treatment"])
        se = float(res.bse["ai_treatment"])
        z = beta / se
        effect = (math.exp(beta) - 1) * 100
        lo = (math.exp(beta - norm.ppf(1-ALPHA/2)*se) - 1) * 100
        hi = (math.exp(beta + norm.ppf(1-ALPHA/2)*se) - 1) * 100
        p_two = float(2 * norm.sf(abs(z)))
        # H1 direcional original: beta < 0 (IA reduz o tempo).
        p_one_reduction = float(norm.cdf(z))
        rows.append({"desfecho": label, "erros_padrao": name, "n": len(d),
                     "beta_IA": beta, "efeito_percentual": effect,
                     "IC95_inf_percentual": lo, "IC95_sup_percentual": hi,
                     "p_bilateral_normal": p_two,
                     "p_unilateral_H1_reducao_normal": p_one_reduction})
        print(f"{name}: efeito={effect:.2f}% | IC95%=[{lo:.2f}%, {hi:.2f}%] | "
              f"p bilateral normal={p_two:.5f} | p unilateral H1 redução={p_one_reduction:.5f}")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Base válida: {len(df)} tarefas; {df['dev_id'].nunique()} desenvolvedores")
    print("Contagem por condição:")
    print(df["ai_treatment"].value_counts().sort_index().rename(index={0: "Sem IA", 1: "IA permitida"}))

    # Descritivas do desfecho principal (planned) e da covariável de ajuste.
    desc_primary = df.groupby("ai_treatment").agg(
        n=("issue_id", "count"),
        tempo_inicial_media=("initial_implementation_time", "mean"),
        tempo_inicial_dp=("initial_implementation_time", "std"),
        tempo_inicial_mediana=("initial_implementation_time", "median"),
        previsao_sem_IA_media=("predicted_time_no_ai", "mean"),
        previsao_sem_IA_dp=("predicted_time_no_ai", "std"),
    ).rename(index={0: "IA nao permitida", 1: "IA permitida"})
    desc_primary.to_csv(OUT / "descritivas_desfecho_primario.csv", encoding="utf-8-sig")
    print("\nDescritivas do desfecho primário (minutos):")
    print(desc_primary.round(2).to_string())

    # Primary analysis: preserve the outcome specified in Marco 2.
    primary_results = fit_and_report(df, "initial_implementation_time", "Tempo inicial (primario - Marco 2)")

    # Official METR replication outcome (secondary because it differs from Marco 2 outcome).
    post_means = df.groupby("ai_treatment")["post_review_implementation_time"].mean()
    missing_post = int(df["post_review_implementation_time"].isna().sum())
    total = df.copy()
    total["post_review_implementation_time"] = total["post_review_implementation_time"].fillna(
        total["ai_treatment"].map(post_means)
    )
    total["total_implementation_time"] = (
        total["initial_implementation_time"] + total["post_review_implementation_time"]
    )
    desc_total = total.groupby("ai_treatment").agg(
        n=("issue_id", "count"),
        total_media=("total_implementation_time", "mean"),
        total_dp=("total_implementation_time", "std"),
        total_mediana=("total_implementation_time", "median"),
        pos_revisao_media=("post_review_implementation_time", "mean"),
        pos_revisao_dp=("post_review_implementation_time", "std"),
    ).rename(index={0: "IA nao permitida", 1: "IA permitida"})
    desc_total.to_csv(OUT / "descritivas_tempo_total_secundario.csv", encoding="utf-8-sig")
    print(f"\nValores pós-revisão ausentes imputados: {missing_post}")
    print("Descritivas do tempo total (minutos):")
    print(desc_total.round(2).to_string())
    secondary_results = fit_and_report(total, "total_implementation_time", "Tempo total (secundario - replicacao METR)")

    all_results = pd.concat([primary_results, secondary_results], ignore_index=True)
    all_results.to_csv(OUT / "resultados_modelos.csv", index=False, encoding="utf-8-sig")

    # Figure: primary (initial time) and secondary (total time), HC3 intervals.
    hc3 = all_results[all_results["erros_padrao"] == "HC3"].copy()
    fig, ax = plt.subplots(figsize=(3.05, 1.58))
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    y = list(range(len(hc3)))
    for i, (_, row) in enumerate(hc3.iterrows()):
        x = row["efeito_percentual"]
        lo = row["IC95_inf_percentual"]
        hi = row["IC95_sup_percentual"]
        ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", capsize=2.5, markersize=3.8, linewidth=1.2)
    ax.set_yticks(y, ["Primário: inicial", "Secundário: total"])
    ax.set_xlim(-2, 44)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Variação ajustada do tempo (%)", fontsize=7.0, labelpad=1.5)
    ax.set_title("Efeito da permissão de IA (IC95%, HC3)", fontsize=7.3, pad=2.5)
    ax.tick_params(axis="both", labelsize=6.4, length=2, pad=1.5)
    ax.grid(axis="x", alpha=.22, linewidth=.45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=.36, right=.99, top=.80, bottom=.28)
    fig.savefig(OUT / "figura_efeitos_HC3.png", dpi=420, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("\nArquivos gerados em resultados/:")
    for p in sorted(OUT.iterdir()):
        print(" -", p)
    print("\nATENÇÃO: o resultado do tempo inicial é o desfecho planejado no Marco 2."
          " Usem esse resultado como análise principal após conferir o protocolo."
          " O tempo total é uma reprodução secundária da análise publicada pela METR.")

if __name__ == "__main__":
    main()
