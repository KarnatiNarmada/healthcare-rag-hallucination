"""Healthcare RAG hallucination study: demo + results app (Streamlit).

Run from the repo root:
    python -m pip install streamlit        (once)
    python -m streamlit run app/app.py

Page 1 "Ask a question"  : type a drug question, pick a version (C0 to C6), see the answer and how it was checked.
                           Needs Ollama running with llama3.2:3b (the same setup as the experiments).
Page 2 "Results"         : charts and tables from stats/ and grades/ (made by eval/stats.py and eval/grade.py).
                           Works without Ollama.
"""
import os, sys, csv, html

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

NAVY, TEAL, CREAM, AMBER, GREY = "#14243A", "#0F766E", "#F6F5F0", "#8F5200", "#64748B"
VERSIONS = {
    "C0": "C0 · Plain answer (baseline)",
    "C1": "C1 · Quote the label",
    "C2": "C2 · Second AI checks",
    "C3": "C3 · Weak-evidence cut-off",
    "C4": "C4 · Checking questions",
    "C5": "C5 · All combined",
    "C6": "C6 · Combined + risk gate",
}
SHORT = {"C0": "C0 Plain", "C1": "C1 Quote", "C2": "C2 2nd AI", "C3": "C3 Cut-off",
         "C4": "C4 Checks", "C5": "C5 Combined", "C6": "C6 Risk gate"}
MODEL, K = "llama3.2:3b", 5

st.set_page_config(page_title="Healthcare RAG · Hallucination Study", page_icon="💊", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background:{CREAM}; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility:hidden; }}
  .block-container {{ padding-top:1.5rem; max-width:1180px; }}
  h1,h2,h3 {{ font-family:Georgia,'Source Serif 4',serif; color:{NAVY}; font-weight:600; }}
  [data-testid="stSidebar"] {{ background:{NAVY}; }}
  [data-testid="stSidebar"] * {{ color:#E8ECF1 !important; }}
  .hero {{ background:{NAVY}; color:#fff; border-radius:16px; padding:28px 34px; margin-bottom:22px; }}
  .hero h1 {{ color:#fff; margin:0 0 6px 0; font-size:2rem; line-height:1.2; }}
  .hero p {{ color:#BFD0E0; margin:0; font-size:1rem; }}
  .kpi {{ background:#FDFCF9; border:1px solid #DCD9CE; border-radius:14px; padding:16px 20px; }}
  .kpi .v {{ font-size:2rem; font-weight:700; color:{NAVY}; line-height:1.1; }}
  .kpi .l {{ font-size:.85rem; color:{GREY}; margin-top:4px; }}
  .kpi.good .v {{ color:{TEAL}; }}
  .kpi.bad .v {{ color:{AMBER}; }}
  .answer {{ border-radius:14px; padding:18px 22px; font-size:1.05rem; line-height:1.55; border:1px solid; }}
  .answer.ok {{ background:#E7F4F1; border-color:#9FD3C9; color:{NAVY}; }}
  .answer.none {{ background:#EAF0F8; border-color:#B7C8DE; color:{NAVY}; }}
  .answer.esc {{ background:#FFF3DC; border-color:#E8C887; color:#5A3A00; }}
  .badge {{ display:inline-block; font-size:.78rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
            padding:3px 10px; border-radius:99px; margin-bottom:10px; }}
  .answer.ok .badge {{ background:{TEAL}; color:#fff; }}
  .answer.none .badge {{ background:#3B5B87; color:#fff; }}
  .answer.esc .badge {{ background:{AMBER}; color:#fff; }}
  .note {{ color:{GREY}; font-size:.88rem; }}
  .src {{ font-family:ui-monospace,Menlo,monospace; font-size:.82rem; background:#fff; border:1px solid #DCD9CE;
          border-radius:8px; padding:6px 10px; margin:4px 0; display:flex; justify-content:space-between; }}
  div.stButton > button[kind="primary"] {{ background:{TEAL}; border:none; border-radius:10px; padding:.55rem 1.4rem; }}
  button[data-baseweb="tab"] {{ font-weight:600; }}
  div.stButton > button[kind="primary"]:disabled {{ background:#B8C2CC; color:#fff; opacity:1; }}
</style>
""", unsafe_allow_html=True)


def path(*p):
    return os.path.join(ROOT, *p)


def kpi(value, label, kind=""):
    return f'<div class="kpi {kind}"><div class="v">{value}</div><div class="l">{label}</div></div>'


@st.cache_resource
def get_retriever():
    from retrieve import Retriever
    return Retriever()


@st.cache_data
def drug_list():
    f = path("questions", "questions_all.csv")
    if not os.path.exists(f):
        return []
    rows = csv.DictReader(open(f, newline="", encoding="utf-8-sig"))
    return sorted({r["drug"] for r in rows if r["drug"].strip()})


def run_version(cond, question, drug):
    from conditions import answer_with_condition
    from generate import answer_question
    retriever = get_retriever()
    if cond in ("C0", "C1"):
        return answer_question(question, retriever, {"C0": "plain", "C1": "quote"}[cond], K, MODEL, drug)
    return answer_with_condition(cond, question, retriever, K, MODEL, drug)


# ----------------------------------------------------------------- page 1
def page_ask():
    st.markdown("""<div class="hero"><h1>Ask a drug-label question</h1>
    <p>Answers come only from 20 openFDA drug labels. Pick a version of the system to see how each safeguard behaves.</p></div>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    drug = c1.selectbox("Drug", ["(search all drugs)"] + drug_list())
    cond = c2.selectbox("System version", list(VERSIONS), index=1, format_func=lambda x: VERSIONS[x])
    question = st.text_area("Your question", height=90,
                            placeholder="e.g. What is the recommended starting dose of sertraline for adults with depression?")
    st.markdown('<p class="note">Research prototype for a university capstone. Not medical advice. '
                'Time per answer on a laptop: C0, C1, C3 about 5 to 15 seconds; C2 and C4 about 30 seconds; '
                'C5 and C6 about 1 to 2 minutes.</p>', unsafe_allow_html=True)
    if not st.button("Get answer", type="primary", disabled=not question.strip()):
        return
    try:
        with st.spinner("Searching the label and checking the answer. This can take a minute..."):
            res = run_version(cond, question.strip(), None if drug.startswith("(") else drug)
    except Exception as e:
        st.error("Could not run the model. Is Ollama running with llama3.2:3b? "
                 "Start it with `ollama serve`, then try again.")
        st.exception(e)
        return
    ans = res["answer"]
    extra = res.get("extra") or {}
    if "clinician review" in ans.lower():
        kind, badge = "esc", "Escalated to a clinician"
    elif "INSUFFICIENT EVIDENCE" in ans.upper():
        kind, badge = "none", "No answer: not enough evidence in the label"
    else:
        kind, badge = "ok", "Answer"
    st.markdown(f'<div class="answer {kind}"><span class="badge">{badge}</span><br>{html.escape(ans)}</div>',
                unsafe_allow_html=True)
    st.write("")
    top = res.get("top_score")
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi(f"{top:.3f}" if top is not None else "-", "Best match score"), unsafe_allow_html=True)
    k2.markdown(kpi(extra.get("severity_predicted", "n/a"), "Risk level (guessed by the system)"), unsafe_allow_html=True)
    k3.markdown(kpi(extra.get("stopped", "none").replace("_", " "), "Stopped early because"), unsafe_allow_html=True)

    st.subheader("Label sources used")
    for s in res.get("sources", []):
        st.markdown(f'<div class="src"><span>{html.escape(s["chunk_id"])}</span><span>score {s["score"]}</span></div>',
                    unsafe_allow_html=True)
    if extra:
        with st.expander("How the answer was checked (all steps)"):
            if "draft" in extra:
                st.markdown("**First draft**"); st.write(extra["draft"])
            if extra.get("checks"):
                st.markdown("**Checking questions**")
                for c in extra["checks"]:
                    st.write(f"- {c['question']}  →  {c['answer']}")
            if "revised" in extra:
                st.markdown("**After checking**"); st.write(extra["revised"])
            if extra.get("verdicts"):
                st.markdown("**Second AI, claim by claim**")
                for v in extra["verdicts"]:
                    st.write(("✅ " if v["supported"] else "❌ ") + v["claim"])


# ----------------------------------------------------------------- page 2
def read_csv(*p):
    f = path(*p)
    return pd.read_csv(f) if os.path.exists(f) else None


def page_results():
    st.markdown("""<div class="hero"><h1>How well does each safeguard work?</h1>
    <p>Seven versions of the system, the same 200 questions, graded by an AI judge. The human check is in progress.</p></div>""",
                unsafe_allow_html=True)
    summ = read_csv("stats", "summary.csv")
    if summ is None:
        st.warning("No results found. Run `python eval/stats.py` from the repo root first.")
        return

    import altair as alt
    s = summ.copy()
    s["rate"] = s["hallucinated_rate"] * 100
    s["lo"] = s["hallucinated_ci_low"] * 100
    s["hi"] = s["hallucinated_ci_high"] * 100
    s["name"] = s["condition"].map(SHORT)
    base, best = s.iloc[0], s.loc[s["rate"].idxmin()]
    k = st.columns(4)
    k[0].markdown(kpi(f"{int(s['n'].max())}", "questions per version"), unsafe_allow_html=True)
    k[1].markdown(kpi(f"{base['rate']:.0f}%", f"hallucinated with the plain baseline ({base['condition']})", "bad"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi(f"{best['rate']:.0f}%", f"lowest rate, version {best['condition']}", "good"), unsafe_allow_html=True)
    k[3].markdown(kpi(f"{len(s)}", "versions compared"), unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["Hallucination by version", "By risk level", "Which differences are real",
                                      "One question, all versions"])

    with tab1:
        order = list(s["name"])
        color = alt.condition(alt.datum.condition == "C0", alt.value(GREY), alt.value(TEAL))
        bars = alt.Chart(s).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=54).encode(
            x=alt.X("name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("rate:Q", title="Hallucinated answers (%)", scale=alt.Scale(domain=[0, max(50, s["hi"].max() + 6)])),
            color=color,
            tooltip=["name", alt.Tooltip("rate:Q", title="%", format=".1f"),
                     alt.Tooltip("lo:Q", title="95% low", format=".1f"), alt.Tooltip("hi:Q", title="95% high", format=".1f")])
        err = alt.Chart(s).mark_errorbar(color=NAVY, thickness=2, ticks={"size": 14, "color": NAVY}).encode(
            x=alt.X("name:N", sort=order), y=alt.Y("lo:Q", title=""), y2="hi:Q")
        labels = alt.Chart(s).mark_text(dy=-26, fontSize=14, fontWeight="bold", color=NAVY).encode(
            x=alt.X("name:N", sort=order), y="hi:Q", text=alt.Text("rate:Q", format=".0f"))
        chart = (bars + err + labels).properties(height=380).configure_view(strokeWidth=0).configure(background="transparent")
        st.altair_chart(chart, width="stretch")
        st.markdown('<p class="note">Bars show the share of answers graded “hallucinated”. Whiskers are 95% confidence ranges: '
                    'if two whiskers overlap a lot, the difference may be chance.</p>', unsafe_allow_html=True)
        grade_cols = [c for c in ["correct", "correct_refusal", "incomplete", "declined", "hallucinated",
                                  "over_refusal", "refused_partial"] if c in summ.columns]
        st.dataframe(summ[["condition", "n"] + grade_cols].rename(columns={"condition": "version"}), hide_index=True)

    with tab2:
        g = read_csv("stats", "by_group.csv")
        if g is None:
            st.info("by_group.csv not found.")
        else:
            split = st.radio("Split by", ["severity_gold", "question_type", "evidence_case"], horizontal=True,
                             format_func=lambda x: {"severity_gold": "Risk level", "question_type": "Question type",
                                                    "evidence_case": "Evidence type"}[x])
            t = g[g["split_by"] == split].pivot(index="group", columns="condition", values="hallucinated_rate") * 100
            st.markdown("**Hallucinated answers (%)**")
            st.dataframe(t.round(1).style.background_gradient(cmap="Oranges", axis=None).format("{:.1f}"))
            st.markdown('<p class="note">Each cell rests on roughly 40 to 80 questions, so small gaps may be chance.</p>',
                        unsafe_allow_html=True)

    with tab3:
        p = read_csv("stats", "pairwise.csv")
        if p is None:
            st.info("pairwise.csv not found.")
        else:
            m = st.radio("Measure", ["hallucinated", "correct"], horizontal=True)
            p = p[p["measure"] == m].copy()
            p["different?"] = p["significant_0.05"].map({True: "YES", False: "no"})
            p["rate_A"] = (p["rate_A"] * 100).round(1)
            p["rate_B"] = (p["rate_B"] * 100).round(1)
            p["p_holm"] = p["p_holm"].round(4)
            p = p.rename(columns={"A": "version A", "B": "version B", "rate_A": "A (%)", "rate_B": "B (%)",
                                  "p_holm": "p-value"})
            st.dataframe(p[["version A", "version B", "A (%)", "B (%)", "p-value", "different?"]], hide_index=True)
            st.markdown('<p class="note">McNemar test on the same questions, Holm-corrected for many comparisons. '
                        '“YES” means p &lt; 0.05.</p>', unsafe_allow_html=True)

    with tab4:
        gr = read_csv("grades", "graded.csv")
        if gr is None:
            st.info("grades/graded.csv not found.")
        else:
            qid = st.selectbox("Question", sorted(gr["q_id"].unique()))
            sub = gr[gr["q_id"] == qid].sort_values("condition")
            first = sub.iloc[0]
            st.markdown(f"**{first['question_type']}** · {first['evidence_case']} · risk: **{first['severity_gold']}**")
            for _, r in sub.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{SHORT.get(r['condition'], r['condition'])}** · grade: `{r['grade']}`")
                    st.write(r["answer"] if isinstance(r["answer"], str) else "")


# ----------------------------------------------------------------- main
st.sidebar.markdown("### 💊 Healthcare RAG")
st.sidebar.caption("Hallucination mitigation on FDA drug labels")
page = st.sidebar.radio("Page", ["Ask a question", "Results"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("CS 69099 Master's Capstone · Kent State University")
if page == "Ask a question":
    page_ask()
else:
    page_results()