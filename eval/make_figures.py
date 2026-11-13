"""Make report-ready figures (PNG, 200 dpi) from the graded answers.

Reads  grades/graded.csv   (and stats/similarity.csv if it exists)
Writes docs/figures/fig1 ... fig6 .png
Run from the repo root:  python eval/make_figures.py
"""
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GREY, AMBER, RED = "#14243A", "#0F766E", "#64748B", "#B7791F", "#B42318"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA5B1", "axes.titleweight": "bold"})

d = pd.read_csv("grades/graded.csv", encoding="utf-8-sig")
d["hall"] = (d["grade"] == "hallucinated").astype(int)
d["refused"] = d["abstained"].astype(str).str.lower().eq("true")
ORDER = [c for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C6G"] if c in set(d["condition"])]
NAME = {"C0": "C0\nPlain", "C1": "C1\nQuote", "C2": "C2\n2nd AI", "C3": "C3\nCut-off", "C4": "C4\nChecks",
        "C5": "C5\nCombined", "C6": "C6\nRisk gate", "C6G": "C6G\nGold risk"}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ---- Figure 1: hallucination rate per version
g = d.groupby("condition")["hall"].agg(["sum", "count"]).loc[ORDER]
pct = 100 * g["sum"] / g["count"]
lo = [100 * wilson(int(k), int(n))[0] for k, n in zip(g["sum"], g["count"])]
hi = [100 * wilson(int(k), int(n))[1] for k, n in zip(g["sum"], g["count"])]
fig, ax = plt.subplots(figsize=(9, 4.6))
cols = [GREY if c == "C0" else TEAL for c in ORDER]
ax.bar(range(len(ORDER)), pct, color=cols, width=0.62)
ax.errorbar(range(len(ORDER)), pct, yerr=[pct - lo, [h - p for h, p in zip(hi, pct)]], fmt="none", ecolor=NAVY, capsize=5, lw=1.6)
for i, (p_, h_) in enumerate(zip(pct, hi)):
    ax.text(i, h_ + 1, f"{p_:.1f}%", ha="center", fontweight="bold", color=NAVY)
ax.set_xticks(range(len(ORDER)), [NAME[c] for c in ORDER])
ax.set_ylabel("Answers graded hallucinated (%)")
ax.set_title("Hallucination rate by version (95% Wilson intervals)", loc="left")
ax.set_ylim(0, max(hi) + 8)
save(fig, "fig1_hallucination_by_version.png")

# ---- Figure 2: by risk level
tiers = [t for t in ["Low", "Moderate", "High", "Critical"] if t in set(d["severity_gold"])]
show = [c for c in ["C0", "C1", "C5", "C6", "C6G"] if c in ORDER]
t = d.pivot_table(index="severity_gold", columns="condition", values="hall", aggfunc="mean").loc[tiers, show] * 100
fig, ax = plt.subplots(figsize=(9, 4.6))
palette = {"C0": GREY, "C1": "#8FB3AE", "C5": TEAL, "C6": NAVY, "C6G": AMBER}
w = 0.8 / len(show)
for j, c in enumerate(show):
    xs = [i + (j - (len(show) - 1) / 2) * w for i in range(len(tiers))]
    ax.bar(xs, t[c], width=w * 0.92, color=palette[c], label=c)
ax.set_xticks(range(len(tiers)), [f"{x}" for x in tiers])
ax.set_xlabel("Risk level of the question (gold label)")
ax.set_ylabel("Answers graded hallucinated (%)")
ax.set_title("Hallucination rate by risk level", loc="left")
ax.legend(frameon=False, ncol=len(show))
save(fig, "fig2_by_risk_level.png")

# ---- Figure 3: High + Critical, benefit and cost
hc = d[d["severity_gold"].isin(["High", "Critical"])]
vers = [c for c in ["C1", "C5", "C6", "C6G"] if c in ORDER]
hallp = [100 * hc[hc.condition == c]["hall"].mean() for c in vers]
refu = []
for c in vers:
    a = hc[(hc.condition == c) & (hc.evidence_case == "answerable")]
    refu.append(100 * a["refused"].mean())
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
axs[0].bar(range(len(vers)), hallp, color=TEAL, width=0.6)
axs[1].bar(range(len(vers)), refu, color=AMBER, width=0.6)
for ax, vals, ttl, yl in [(axs[0], hallp, "Benefit: hallucinated answers", "% of High and Critical questions"),
                          (axs[1], refu, "Cost: answerable questions refused", "% of answerable High and Critical")]:
    ax.set_xticks(range(len(vers)), vers)
    ax.set_title(ttl, loc="left", fontsize=11)
    ax.set_ylabel(yl)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", color=NAVY)
    ax.set_ylim(0, max(vals) * 1.25 + 1)
fig.suptitle("The risk gate: fewer hallucinations, many more refusals", x=0.02, ha="left", fontweight="bold")
save(fig, "fig3_gate_benefit_and_cost.png")

# ---- Figure 4: retrieval score overlap
b = d[d["condition"] == "C0"].dropna(subset=["top_score"])
fig, ax = plt.subplots(figsize=(9, 4.4))
for case, color, lab in [("answerable", TEAL, "Answerable"), ("partial", "#C9A227", "Partial"), ("unanswerable", AMBER, "Unanswerable")]:
    ax.hist(b[b.evidence_case == case]["top_score"], bins=24, alpha=0.65, color=color, label=lab)
ax.axvline(0.71, color=NAVY, ls="--", lw=2)
ax.text(0.712, ax.get_ylim()[1] * 0.92, " threshold 0.71", color=NAVY, fontweight="bold")
ax.set_xlabel("Best retrieval score for the question")
ax.set_ylabel("Questions")
ax.set_title("Scores of answerable and unanswerable questions overlap", loc="left")
ax.legend(frameon=False)
save(fig, "fig4_score_overlap.png")

# ---- Figure 5: error types
h = d[d["hall"] == 1]
ct = pd.crosstab(h["condition"], h["error_type"]).reindex(ORDER).fillna(0)
labels = {"unsupported_claim": "Extra detail not in label", "fabricated_fact": "Made-up fact",
          "contradicts_gold": "Contradicts the label", "answered_unanswerable": "Answered an unanswerable question",
          "invented_missing_part": "Invented the missing part", "wrong_number": "Wrong number", "off_target": "Off target"}
cl = [TEAL, AMBER, NAVY, GREY, "#8FB3AE", RED, "#C9A227"]
fig, ax = plt.subplots(figsize=(9, 4.8))
bottom = [0] * len(ct)
for k, col in zip(ct.columns, cl * 2):
    ax.bar(range(len(ct)), ct[k], bottom=bottom, color=col, width=0.62, label=labels.get(k, k))
    bottom = [x + y for x, y in zip(bottom, ct[k])]
ax.set_xticks(range(len(ORDER)), [NAME[c] for c in ORDER])
ax.set_ylabel("Hallucinated answers")
ax.set_title("What kind of error remains", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")
save(fig, "fig5_error_types.png")

# ---- Figure 6: similarity index versus the judge
sf = "stats/similarity.csv"
if os.path.exists(sf):
    s = pd.read_csv(sf)
    s = s[s["grade"].isin(["correct", "hallucinated"])]
    y = (s["grade"] == "correct").astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    names = {"em": "Exact match", "f1": "Word overlap (F1)", "rougeL": "ROUGE-L", "cos": "Embedding cosine"}
    vals = []
    for k in names:
        rk = s[k].rank()
        vals.append((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(list(names.values()), vals, color=[GREY, TEAL, TEAL, "#8FB3AE"])
    ax.axvline(0.5, color=NAVY, ls="--")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontweight="bold", color=NAVY)
    ax.set_xlim(0.4, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("AUC: separates correct from hallucinated answers (0.5 = chance)")
    ax.set_title("Similarity Index versus the AI judge", loc="left")
    save(fig, "fig6_similarity_vs_judge.png")
else:
    print("skipped fig6: run python eval/similarity.py first")