"""Healthcare RAG hallucination study: demo + results app (Streamlit).

Run from the repo root:
    python -m pip install streamlit        (once)
    python -m streamlit run app/app.py

Page 1 "Ask a question"  : type a drug question, pick a version (C0 to C6), see the answer and how it was checked.
                           Needs Ollama running with llama3.2:3b (the same setup as the experiments).
Page 2 "Results"         : one tab per research question, computed live from grades/graded.csv.
                           Works without Ollama and without the stats/ folder.
"""
import os, sys, csv, html, math, itertools, threading, time

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
         "C4": "C4 Checks", "C5": "C5 Combined", "C6": "C6 Risk gate", "C6G": "C6G Gold risk"}
EVIDENCE = {"answerable": "Answerable (label has the answer)", "partial": "Partial (label has half)",
            "unanswerable": "Unanswerable (not in label)"}
ERR_NAMES = {"unsupported_claim": "Extra detail not in label", "fabricated_fact": "Made-up fact",
             "contradicts_gold": "Contradicts the label", "answered_unanswerable": "Answered an unanswerable question",
             "invented_missing_part": "Invented the missing half", "wrong_number": "Wrong number", "off_target": "Off target"}
MODEL, K = "llama3.2:3b", 5

st.set_page_config(page_title="Healthcare RAG · Hallucination Study", page_icon="💊", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:{CREAM}; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ visibility:hidden; }}
  [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ visibility:visible !important; }}
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
  .finding {{ background:#E7F4F1; border-left:5px solid {TEAL}; border-radius:8px; padding:12px 18px; margin:6px 0 14px 0;
              color:{NAVY}; font-size:1rem; }}
  .src {{ font-family:ui-monospace,Menlo,monospace; font-size:.82rem; background:#fff; border:1px solid #DCD9CE;
          border-radius:8px; padding:6px 10px; margin:4px 0; display:flex; justify-content:space-between; }}
  div.stButton > button[kind="primary"] {{ background:{TEAL}; border:none; border-radius:10px; padding:.55rem 1.4rem; }}
  div.stButton > button[kind="primary"]:disabled {{ background:#B8C2CC; color:#fff; opacity:1; }}
  button[data-baseweb="tab"] {{ font-weight:600; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{ background:#FDFCF9; border-radius:14px; border-color:#DCD9CE !important; }}
  .strip {{ display:flex; gap:14px; margin-bottom:18px; }}
  .strip .kpi {{ flex:1; }}
  .chip {{ display:inline-block; font-size:.78rem; font-weight:700; padding:3px 12px; border-radius:99px; color:#fff; }}
  .chip.Low {{ background:{TEAL}; }} .chip.Moderate {{ background:#3B5B87; }}
  .chip.High {{ background:{AMBER}; }} .chip.Critical {{ background:#B42318; }} .chip.na {{ background:{GREY}; }}
  .gauge {{ position:relative; height:12px; border-radius:99px; background:#E4E7EB; margin:10px 0 4px 0; }}
  .gauge .fill {{ height:12px; border-radius:99px; background:{TEAL}; }}
  .gauge .mark {{ position:absolute; top:-4px; width:2px; height:20px; background:{NAVY}; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .dot.ready {{ background:#2E9E6B; }} .dot.busy {{ background:#E0A100; }}
  .panelrow {{ display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #ECE9DF; font-size:.95rem; }}
  .panelrow:last-child {{ border-bottom:none; }}
  .panelrow .k {{ color:{GREY}; }}
</style>
""", unsafe_allow_html=True)


def path(*p):
    return os.path.join(ROOT, *p)


def kpi(value, label, kind=""):
    return f'<div class="kpi {kind}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def finding(text):
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


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


@st.cache_data
def question_text():
    f = path("questions", "questions_all.csv")
    if not os.path.exists(f):
        return {}
    return {r["q_id"]: r["question"] for r in csv.DictReader(open(f, newline="", encoding="utf-8-sig"))}


def run_version(cond, question, drug):
    from conditions import answer_with_condition
    from generate import answer_question
    retriever = get_retriever()
    if cond in ("C0", "C1"):
        return answer_question(question, retriever, {"C0": "plain", "C1": "quote"}[cond], K, MODEL, drug)
    return answer_with_condition(cond, question, retriever, K, MODEL, drug)


# ----------------------------------------------------------------- serving layer (shared by all visitors)
MAX_PER_SESSION = 30
TIME = {"C0": "5 to 15 s", "C1": "5 to 15 s", "C2": "about 30 s", "C3": "5 to 15 s", "C4": "about 30 s",
        "C5": "1 to 2 min", "C6": "1 to 2 min"}
DESC = {"C0": "The model answers from the five best label passages. No safeguards. This is the baseline.",
        "C1": "Every statement must carry an exact quote from the label, or the model refuses.",
        "C2": "A second AI checks each sentence against the label and drops unsupported ones.",
        "C3": "Refuses if the best passage is a weak match; otherwise answers like C0.",
        "C4": "The model writes checking questions about its own draft and revises it.",
        "C5": "All safeguards in sequence: evidence cut-off, quoting, checking questions, second AI.",
        "C6": "C5 plus a risk gate: dangerous questions (dosing, interactions) get stricter checks and escalate."}
EXAMPLES = {"Side effects": ("warfarin", "What are the most common serious side effects of warfarin?"),
            "Dosing": ("furosemide", "What is the usual starting dose of furosemide tablets for edema in adults?"),
            "Interaction": ("warfarin", "Can I take warfarin together with aspirin?"),
            "Not in label": ("amoxicillin", "What is the best brand of amoxicillin to buy online?")}


@st.cache_resource
def service():
    """One object shared by every browser session: a lock so the single local model handles one
    question at a time, a queue counter, simple counters, and a small in-memory answer cache.
    Answers are deterministic (temperature 0, fixed seed), so a repeated question can reuse the result.
    Nothing is written to disk."""
    return {"lock": threading.Lock(), "mu": threading.Lock(), "waiting": 0, "busy": False,
            "served": 0, "cache": {}}


def serve(cond, question, drug, status):
    S = service()
    key = (cond, question.strip().lower(), drug)
    with S["mu"]:
        hit = S["cache"].get(key)
        if hit is not None:
            return hit, 0.0, True
        ahead = S["waiting"] + (1 if S["busy"] else 0)
        S["waiting"] += 1
    if ahead:
        status.update(label=f"Waiting for the model: {ahead} request(s) ahead of you")
    with S["lock"]:
        with S["mu"]:
            S["waiting"] -= 1
            S["busy"] = True
        status.update(label="Searching the label and checking the answer...")
        t0 = time.time()
        try:
            res = run_version(cond, question, drug)
        finally:
            with S["mu"]:
                S["busy"] = False
        dt = time.time() - t0
    with S["mu"]:
        S["served"] += 1
        S["cache"][key] = res
        while len(S["cache"]) > 300:
            S["cache"].pop(next(iter(S["cache"])))
    return res, dt, False


def use_example(drug, q):
    st.session_state["q"] = q
    if drug in drug_list():
        st.session_state["drug"] = drug


def system_status():
    S = service()
    with S["mu"]:
        busy, waiting, served = S["busy"], S["waiting"], S["served"]
    return busy, waiting, served


# ----------------------------------------------------------------- page 1
def page_ask():
    st.markdown("""<div class="hero"><h1>Drug-label question answering</h1>
    <p>Answers come only from 20 openFDA drug labels. Research prototype, not medical advice. Do not enter patient information.</p></div>""",
                unsafe_allow_html=True)
    busy, waiting, served = system_status()
    cond_now = st.session_state.get("cond", "C1")
    state = f'<span class="dot {"busy" if busy else "ready"}"></span>{"Busy, " + str(waiting) + " waiting" if busy else "Ready"}'
    st.markdown(f'<div class="strip">{kpi("20", "drug labels")}{kpi("444", "label passages searched")}'
                f'{kpi(TIME.get(cond_now, "-"), f"typical time for {cond_now}")}{kpi(state, f"model status, {served} answered")}</div>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Your question")
            c1, c2 = st.columns(2)
            drug = c1.selectbox("Drug", ["(search all drugs)"] + drug_list(), key="drug")
            cond = c2.selectbox("System version", list(VERSIONS), index=1, format_func=lambda x: VERSIONS[x], key="cond")
            st.caption(DESC[cond])
            st.caption("Try an example")
            ex = st.columns(len(EXAMPLES))
            for col, (label, (d_, q_)) in zip(ex, EXAMPLES.items()):
                col.button(label, key="ex_" + label, on_click=use_example, args=(d_, q_), use_container_width=True)
            question = st.text_area("Question", height=90, key="q", label_visibility="collapsed",
                                    placeholder="e.g. What is the recommended starting dose of sertraline for adults with depression?")
            asked = st.session_state.get("asked", 0)
            go = st.button("Get answer", type="primary", disabled=not question.strip())
            if go and asked >= MAX_PER_SESSION:
                st.warning("This session reached its limit of %d questions. Reload the page to continue." % MAX_PER_SESSION)
                go = False
            if go:
                try:
                    with st.status("Starting...", expanded=False) as status:
                        res, dt, cached = serve(cond, question.strip(), None if drug.startswith("(") else drug, status)
                        status.update(label="Done" + (" (reused an earlier identical answer)" if cached else f" in {dt:.1f} s"),
                                      state="complete")
                    st.session_state["asked"] = asked + 1
                    st.session_state["last"] = {"res": res, "dt": dt, "cached": cached, "cond": cond, "question": question.strip()}
                    st.rerun()
                except Exception as e:
                    st.error("Could not run the model. Is Ollama running with llama3.2:3b? "
                             "Start it with `ollama serve`, then try again.")
                    st.exception(e)

    last = st.session_state.get("last")
    with right:
        with st.container(border=True):
            st.markdown("#### Safety panel")
            if not last:
                st.caption("After you ask a question, this panel shows the risk level, how strong the evidence was, "
                           "and whether a safeguard stopped the answer.")
            else:
                res, extra = last["res"], last["res"].get("extra") or {}
                tier = extra.get("severity_predicted") or "Not used"
                top = res.get("top_score")
                stopped = str(extra.get("stopped", "none")).replace("_", " ")
                chip = f'<span class="chip {tier if tier in ("Low", "Moderate", "High", "Critical") else "na"}">{html.escape(str(tier))}</span>'
                thr = 0.73 if tier in ("High", "Critical") else 0.71
                gauge = ""
                if top is not None:
                    gauge = (f'<div class="gauge"><div class="fill" style="width:{max(0, min(1, top)) * 100:.0f}%"></div>'
                             f'<div class="mark" style="left:{thr * 100:.0f}%"></div></div>')
                tm = "instant (reused)" if last["cached"] else f'{last["dt"]:.1f} s'
                rows_ = (f'<div class="panelrow"><span class="k">Version</span><span>{html.escape(SHORT.get(last["cond"], last["cond"]))}</span></div>'
                         f'<div class="panelrow"><span class="k">Risk level (system guess)</span>{chip}</div>')
                if top is not None:
                    rows_ += f'<div class="panelrow"><span class="k">Best match score</span><span><b>{top:.3f}</b></span></div>'
                st.markdown(rows_, unsafe_allow_html=True)
                if gauge:
                    st.markdown(gauge, unsafe_allow_html=True)
                    st.caption(f"The line marks the evidence threshold ({thr:.2f}). A score below it means the label is a weak match.")
                st.markdown(f'<div class="panelrow"><span class="k">Stopped early because</span><span>{html.escape(stopped)}</span></div>'
                            f'<div class="panelrow"><span class="k">Time</span><span>{tm}</span></div>', unsafe_allow_html=True)

    if not last:
        return
    res, extra = last["res"], last["res"].get("extra") or {}
    ans = res["answer"]
    if "clinician review" in ans.lower():
        kind, badge = "esc", "Escalated to a clinician"
    elif "INSUFFICIENT EVIDENCE" in ans.upper():
        kind, badge = "none", "No answer: not enough evidence in the label"
    else:
        kind, badge = "ok", "Answer"
    with st.container(border=True):
        st.markdown(f"#### Answer")
        st.caption(last["question"])
        st.markdown(f'<div class="answer {kind}"><span class="badge">{badge}</span><br>{html.escape(ans)}</div>',
                    unsafe_allow_html=True)
    c_src, c_chk = st.columns(2, gap="large")
    with c_src:
        with st.container(border=True):
            st.markdown("#### Label passages used")
            for i, s_ in enumerate(res.get("sources", [])):
                st.markdown(f'<div class="src"><span>{html.escape(s_["chunk_id"])}</span><span>score {s_["score"]}</span></div>',
                            unsafe_allow_html=True)
                if s_.get("text"):
                    with st.expander("Show passage"):
                        st.write(s_["text"])
    with c_chk:
        with st.container(border=True):
            st.markdown("#### How the answer was checked")
            if not extra or not any(k in extra for k in ("draft", "checks", "revised", "verdicts")):
                st.caption("This version has no extra checking steps.")
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


# ----------------------------------------------------------------- statistics helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, run = [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


@st.cache_data
def load_graded():
    f = path("grades", "graded.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["hall"] = (d["grade"] == "hallucinated").astype(int)
    d["good"] = d["grade"].isin(["correct", "correct_refusal"]).astype(int)
    return d


@st.cache_data
def pairwise(d):
    conds = sorted(d["condition"].unique())
    w = d.pivot_table(index="q_id", columns="condition", values="hall", aggfunc="first")
    rows = []
    for a, b in itertools.combinations(conds, 2):
        t = w[[a, b]].dropna()
        x = int(((t[a] == 1) & (t[b] == 0)).sum()); y = int(((t[a] == 0) & (t[b] == 1)).sum())
        rows.append({"A": a, "B": b, "A (%)": round(100 * t[a].mean(), 1), "B (%)": round(100 * t[b].mean(), 1),
                     "p": mcnemar_exact(x, y)})
    adj = holm([r["p"] for r in rows])
    for r, p in zip(rows, adj):
        r["p-value (Holm)"] = round(p, 4)
        r["different?"] = "YES" if p < 0.05 else "no"
        del r["p"]
    return pd.DataFrame(rows)


def download(df, name, label="Download this table (CSV)"):
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=name, mime="text/csv")


# ----------------------------------------------------------------- page 2
def page_results():
    import altair as alt
    st.markdown("""<div class="hero"><h1>Results by research question</h1>
    <p>Eight versions of the system, the same 200 questions, graded by an AI judge that was checked against a human on 105 answers.
    C6G is a diagnostic: the risk gate with the true risk level instead of the system's guess.</p></div>""",
                unsafe_allow_html=True)
    d = load_graded()
    if d is None:
        st.warning("grades/graded.csv not found. Run `python eval/grade.py` first.")
        return
    conds = sorted(d["condition"].unique())
    rate = d.groupby("condition")["hall"].agg(["sum", "count"])
    rate["pct"] = 100 * rate["sum"] / rate["count"]
    base, best = rate.loc["C0"], rate["pct"].idxmin()
    c1r = rate.loc["C1"] if "C1" in rate.index else None
    k = st.columns(5)
    k[0].markdown(kpi(f"{int(rate['count'].max())}", "questions per version"), unsafe_allow_html=True)
    k[1].markdown(kpi(f"{base['pct']:.0f}%", "hallucinated, plain baseline (C0)", "bad"), unsafe_allow_html=True)
    if c1r is not None:
        k[2].markdown(kpi(f"{c1r['pct']:.0f}%", "hallucinated, quoting the label (C1)", "good"), unsafe_allow_html=True)
    k[3].markdown(kpi(f"{rate.loc[best, 'pct']:.0f}%", f"lowest rate of all ({best})", "good"), unsafe_allow_html=True)
    k[4].markdown(kpi("93%", "AI judge agrees with a human (105 answers)"), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["RQ1 · Strategies", "RQ2 · Errors", "RQ3 · Evidence type", "RQ4 · Risk level",
                                              "Cost", "Evidence gate", "Similarity", "One question"])

    # ---- RQ1
    with t1:
        s = rate.reset_index()
        s["name"] = s["condition"].map(SHORT)
        s["lo"] = [100 * wilson(int(a), int(b))[0] for a, b in zip(s["sum"], s["count"])]
        s["hi"] = [100 * wilson(int(a), int(b))[1] for a, b in zip(s["sum"], s["count"])]
        order = list(s["name"])
        color = alt.condition(alt.datum.condition == "C0", alt.value(GREY), alt.value(TEAL))
        bars = alt.Chart(s).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=54).encode(
            x=alt.X("name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("pct:Q", title="Hallucinated answers (%)", scale=alt.Scale(domain=[0, max(50, s["hi"].max() + 6)])),
            color=color, tooltip=["name", alt.Tooltip("pct:Q", title="%", format=".1f"),
                                  alt.Tooltip("lo:Q", title="95% low", format=".1f"), alt.Tooltip("hi:Q", title="95% high", format=".1f")])
        err = alt.Chart(s).mark_errorbar(color=NAVY, thickness=2, ticks={"size": 14, "color": NAVY}).encode(
            x=alt.X("name:N", sort=order), y=alt.Y("lo:Q", title=""), y2="hi:Q")
        lab = alt.Chart(s).mark_text(dy=-26, fontSize=14, fontWeight="bold", color=NAVY).encode(
            x=alt.X("name:N", sort=order), y="hi:Q", text=alt.Text("pct:Q", format=".0f"))
        st.altair_chart((bars + err + lab).properties(height=380).configure_view(strokeWidth=0)
                        .configure(background="transparent"), width="stretch")
        finding("<b>Quoting the label does most of the work.</b> It cuts hallucinations from 33% to 12%. "
                "Adding more checks (C5, C6, C6G) is not significantly better than quoting alone (see the table below).")
        st.markdown("**Which differences are real?** (same questions, McNemar test, Holm-corrected)")
        pw = pairwise(d)
        ref = st.radio("Show comparisons with", ["All pairs"] + conds, horizontal=True)
        if ref != "All pairs":
            pw = pw[(pw["A"] == ref) | (pw["B"] == ref)]
        st.dataframe(pw, hide_index=True)
        download(pw, "pairwise.csv")

    # ---- RQ2
    with t2:
        h = d[d["hall"] == 1].copy()
        h["Error type"] = h["error_type"].map(ERR_NAMES).fillna(h["error_type"])
        cnt = h.groupby(["condition", "Error type"]).size().reset_index(name="answers")
        ch = alt.Chart(cnt).mark_bar().encode(
            x=alt.X("condition:N", title="Version", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("answers:Q", title="Hallucinated answers"),
            color=alt.Color("Error type:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["condition", "Error type", "answers"]).properties(height=380).configure(background="transparent")
        st.altair_chart(ch, width="stretch")
        finding("<b>The most common error is an extra detail that the label does not support</b>, not a wrong number. "
                "Wrong numbers are rare, which matters for dosing questions.")
        tab = pd.crosstab(h["Error type"], h["condition"])
        st.dataframe(tab)
        download(tab, "error_types.csv")
        st.markdown('<p class="note">Error types were assigned by the AI judge; the human grader marked grades only.</p>',
                    unsafe_allow_html=True)

    # ---- RQ3
    with t3:
        ev = d.groupby(["evidence_case", "condition"])["hall"].mean().mul(100).reset_index(name="pct")
        ev["Evidence"] = ev["evidence_case"].map(EVIDENCE)
        ch = alt.Chart(ev).mark_bar(color=TEAL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("condition:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pct:Q", title="Hallucinated answers (%)"),
            tooltip=["condition", alt.Tooltip("pct:Q", format=".1f")]).properties(width=240, height=260)
        st.altair_chart(ch.facet(column=alt.Column("Evidence:N", title=None, sort=list(EVIDENCE.values()))).configure(background="transparent"))
        finding("<b>Partial and unanswerable questions are where safeguards help most</b> (partial: 50% to 10%, "
                "unanswerable: 23% to 5% with quoting). On answerable questions the safeguards help little.")
        u = d[d["evidence_case"] == "unanswerable"].copy()
        u["What the system did"] = u["grade"].map({"correct_refusal": "Refused (INSUFFICIENT EVIDENCE)",
                                                    "declined": "Declined in its own words",
                                                    "hallucinated": "Fabricated an answer"}).fillna("Other")
        st.markdown("**When the label has no answer (60 questions): what did each version do?**")
        ut = pd.crosstab(u["condition"], u["What the system did"])
        st.dataframe(ut)
        download(ut, "unanswerable_outcomes.csv")

    # ---- RQ4
    with t4:
        split = st.radio("Split by", ["severity_gold", "question_type"], horizontal=True,
                         format_func=lambda x: {"severity_gold": "Risk level (Low, Moderate, High, Critical)",
                                                "question_type": "Question type"}[x])
        t = d.pivot_table(index=split, columns="condition", values="hall", aggfunc="mean") * 100
        if split == "severity_gold":
            t = t.reindex([x for x in ["Low", "Moderate", "High", "Critical"] if x in t.index])
        st.markdown("**Hallucinated answers (%)**")
        st.dataframe(t.round(1).style.background_gradient(cmap="Oranges", axis=None).format("{:.1f}"))
        finding("<b>Critical questions improve a lot with quoting (43% to 12%)</b>, and the risk gate trends lower still "
                "(7% with the system's guess, 3% with the true risk level). "
                "Each risk group has only about 40 to 60 questions, so these further gaps may be chance.")
        download(t.round(1), "by_group.csv")

        # ---- the dangerous questions only: does the gate help, and what does it cost?
        st.markdown("---")
        st.markdown("### The dangerous questions only (High and Critical)")
        hi = d[d["severity_gold"].isin(["High", "Critical"])].copy()
        hi["refused"] = hi["abstained"].astype(str).str.lower().eq("true")
        rows = []
        for c in ["C1", "C5", "C6", "C6G"]:
            x = hi[hi["condition"] == c]
            ans = x[x["evidence_case"] == "answerable"]
            rows.append({"Version": SHORT.get(c, c), "Questions": len(x),
                         "Hallucinated": f"{int(x['hall'].sum())} ({100 * x['hall'].mean():.1f}%)",
                         "Answerable questions refused": f"{int(ans['refused'].sum())} of {len(ans)}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        w = hi.pivot_table(index="q_id", columns="condition", values="hall", aggfunc="first")
        prs, raw = [], []
        for a_, b_ in [("C5", "C6"), ("C5", "C6G"), ("C6", "C6G"), ("C1", "C6G")]:
            if a_ in w and b_ in w:
                tt = w[[a_, b_]].dropna()
                x_ = int(((tt[a_] == 1) & (tt[b_] == 0)).sum())
                y_ = int(((tt[a_] == 0) & (tt[b_] == 1)).sum())
                prs.append({"Pair": f"{a_} vs {b_}", "Only first wrong": x_, "Only second wrong": y_})
                raw.append(mcnemar_exact(x_, y_))
        if prs:
            for r_, p_ in zip(prs, holm(raw)):
                r_["p-value (Holm)"] = round(p_, 4)
                r_["different?"] = "YES" if p_ < 0.05 else "no"
            st.dataframe(pd.DataFrame(prs), hide_index=True)
        finding("<b>The gate trends toward fewer hallucinations on dangerous questions, but not significantly.</b> "
                "The best case (C1 against C6G) is borderline after correction. The price is large: the gated versions "
                "refuse far more answerable High and Critical questions than quoting alone.")

    # ---- cost
    with t5:
        a = d[d["evidence_case"] == "answerable"].copy()
        a["Outcome"] = a["grade"].map({"correct": "Correct", "incomplete": "Incomplete", "hallucinated": "Hallucinated",
                                       "over_refusal": "Refused a good question", "declined": "Declined"}).fillna("Other")
        cnt = a.groupby(["condition", "Outcome"]).size().reset_index(name="answers")
        ch = alt.Chart(cnt).mark_bar().encode(
            x=alt.X("condition:N", title="Version", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("answers:Q", title="Answerable questions (of 80)"),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Correct", "Incomplete", "Hallucinated", "Refused a good question", "Declined"],
                                                         range=[TEAL, "#8FB3AE", AMBER, GREY, "#B8C2CC"])),
            order=alt.Order("Outcome:N")).properties(height=380).configure(background="transparent")
        st.altair_chart(ch, width="stretch")
        ref_ = a.assign(refused=a["abstained"].astype(str).str.lower().eq("true")).groupby("condition")["refused"].sum()
        txt = ", ".join(f"{c} {int(ref_.get(c, 0))}" for c in ["C1", "C5", "C6", "C6G"] if c in ref_.index)
        finding(f"<b>Safeguards have a price.</b> Answerable questions refused out of 80 ({txt}). "
                "The combined and gated versions refuse far more questions that the label can answer, "
                "so fewer questions get a correct answer than with quoting alone.")

    # ---- evidence gate: why a score cut-off alone does not work
    with t6:
        import altair as alt
        base_ = d[d["condition"] == "C0"].dropna(subset=["top_score"]).copy()
        base_["Evidence"] = base_["evidence_case"].map(EVIDENCE)
        hist = alt.Chart(base_).mark_bar(opacity=0.75).encode(
            x=alt.X("top_score:Q", bin=alt.Bin(step=0.01), title="Best retrieval score for the question"),
            y=alt.Y("count():Q", stack=None, title="Questions"),
            color=alt.Color("Evidence:N", scale=alt.Scale(domain=list(EVIDENCE.values()), range=[TEAL, "#C9A227", AMBER]))
        ).properties(height=340)
        rule = alt.Chart(pd.DataFrame({"x": [0.71]})).mark_rule(color=NAVY, strokeDash=[6, 4], size=2).encode(x="x:Q")
        st.altair_chart((hist + rule).configure(background="transparent"), width="stretch")
        finding("<b>The scores of answerable and unanswerable questions overlap.</b> A cut-off at 0.71 (dashed line) cannot "
                "tell them apart, which is why the weak-evidence cut-off alone (C3) changed nothing: 33% hallucinated, the same as the baseline.")
        med = base_.groupby("Evidence")["top_score"].agg(["count", "median", "min", "max"]).round(3)
        st.dataframe(med)
        download(med, "score_by_evidence.csv")

    # ---- similarity index
    with t7:
        import altair as alt
        sf = path("stats", "similarity.csv")
        if not os.path.exists(sf):
            st.info("Run `python eval/similarity.py` first to create stats/similarity.csv.")
        else:
            sm = pd.read_csv(sf)
            st.markdown("**How close is each answer to the gold answer?** Refusals are left out (they contain no content).")
            names = {"f1": "Word overlap (F1)", "rougeL": "ROUGE-L", "cos": "Meaning (embedding cosine)"}
            mc = sm.groupby("condition")[["f1", "rougeL", "cos"]].mean().reset_index().melt("condition", var_name="metric", value_name="score")
            mc["Metric"] = mc["metric"].map(names)
            ch = alt.Chart(mc).mark_bar().encode(
                x=alt.X("condition:N", title="Version", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("score:Q", title="Mean score (0 to 1)"),
                xOffset="Metric:N", color=alt.Color("Metric:N", scale=alt.Scale(range=[TEAL, "#8FB3AE", NAVY])),
                tooltip=["condition", "Metric", alt.Tooltip("score:Q", format=".3f")]).properties(height=360).configure(background="transparent")
            st.altair_chart(ch, width="stretch")

            sub_ = sm[sm["grade"].isin(["correct", "hallucinated"])].copy()
            yy = (sub_["grade"] == "correct").astype(int)
            aucs = []
            for k_, nm in [("em", "Exact match"), ("f1", names["f1"]), ("rougeL", names["rougeL"]), ("cos", names["cos"])]:
                rk = sub_[k_].rank()
                n1, n0 = int(yy.sum()), int((1 - yy).sum())
                aucs.append({"Metric": nm, "Separates correct from hallucinated (AUC)": round((rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), 3)})
            st.markdown("**Does the index agree with the AI judge?** 0.5 means no better than chance, 1.0 means perfect.")
            st.dataframe(pd.DataFrame(aucs), hide_index=True)
            finding("<b>Word-overlap scores agree with the judge; meaning-based and exact-match scores do not.</b> "
                    "A wrong answer about the right drug still sits close in meaning, so the index supports the judge but cannot replace it.")
            bg = sm.groupby("grade")[["f1", "rougeL", "cos"]].mean().round(3)
            st.dataframe(bg)
            download(bg, "similarity_by_grade.csv")

    # ---- one question
    with t8:
        qid = st.selectbox("Question", sorted(d["q_id"].unique()))
        sub = d[d["q_id"] == qid].sort_values("condition")
        first = sub.iloc[0]
        st.markdown(f"**{question_text().get(qid, qid)}**")
        st.caption(f"{first['question_type']} · {first['evidence_case']} · risk: {first['severity_gold']}")
        for _, r in sub.iterrows():
            with st.container(border=True):
                st.markdown(f"**{SHORT.get(r['condition'], r['condition'])}** · grade: `{r['grade']}`")
                st.write(r["answer"] if isinstance(r["answer"], str) else "")


# ----------------------------------------------------------------- main
st.sidebar.markdown("### 💊 Healthcare RAG")
st.sidebar.caption("Hallucination mitigation on FDA drug labels")
page = st.sidebar.radio("Page", ["Ask a question", "Results"], label_visibility="collapsed")
_b, _w, _n = system_status()
st.sidebar.markdown("---")
st.sidebar.caption(("Model busy, %d waiting" % _w if _b else "Model ready") + " · %d answered" % _n)
st.sidebar.caption("CS 69099 Master's Capstone · Kent State University")
if page == "Ask a question":
    page_ask()
else:
    page_results()