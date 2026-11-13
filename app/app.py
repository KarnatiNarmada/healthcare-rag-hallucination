"""Drug label assistant (Streamlit).

Run from the repo root:
    python -m streamlit run app/app.py

Pages: Ask (FDA labels or an uploaded document), Evaluation (results computed from grades/graded.csv), Guide.
The Ask page needs Ollama running with llama3.2:3b. Evaluation and Guide work without it.
"""
import os, re, sys, csv, html, math, itertools, threading, time

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

NAVY, TEAL, CREAM, AMBER, GREY = "#14243A", "#0F766E", "#F6F5F0", "#8F5200", "#64748B"
VERSIONS = {
    "C0": "C0 · Baseline (no safeguards)",
    "C1": "C1 · Source-quoted answers",
    "C2": "C2 · Independent claim verification",
    "C3": "C3 · Minimum-evidence threshold",
    "C4": "C4 · Self-verification questions",
    "C5": "C5 · Combined safeguards",
    "C6": "C6 · Combined safeguards with risk-tiered review",
}
SHORT = {"C0": "C0 Plain", "C1": "C1 Quote", "C2": "C2 2nd AI", "C3": "C3 Cut-off",
         "C4": "C4 Checks", "C5": "C5 Combined", "C6": "C6 Risk gate", "C6G": "C6G Gold risk"}
EVIDENCE = {"answerable": "Answerable (label has the answer)", "partial": "Partial (label has half)",
            "unanswerable": "Unanswerable (not in label)"}
ERR_NAMES = {"unsupported_claim": "Extra detail not in label", "fabricated_fact": "Made-up fact",
             "contradicts_gold": "Contradicts the label", "answered_unanswerable": "Answered an unanswerable question",
             "invented_missing_part": "Invented the missing half", "wrong_number": "Wrong number", "off_target": "Off target"}
MODEL, K = "llama3.2:3b", 5

st.set_page_config(page_title="Drug Label Assistant", page_icon="💊", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  html, body, [class*="css"], .stApp {{ font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif; }}
  .stApp {{ background:#F5F6F8; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ visibility:hidden; }}
  [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ visibility:visible !important; }}
  .block-container {{ padding-top:2rem; max-width:920px; }}
  h1,h2,h3,h4 {{ color:#101828; font-weight:650; letter-spacing:-.01em; }}
  h4 {{ font-size:1.02rem; }}
  [data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid #E4E7EC; }}
  [data-testid="stSidebar"] * {{ color:#344054; }}
  .brand {{ font-size:1.55rem; font-weight:800; color:#101828; margin:6px 0 2px 0; line-height:1.15; letter-spacing:-.02em; }}
  .brand .ico {{ font-size:1.7rem; margin-right:4px; }}
  .brandsub {{ font-size:.92rem; color:#667085; margin-bottom:14px; }}
  .pagehead h1 {{ font-size:1.7rem; margin:0 0 4px 0; }}
  .pagehead p {{ color:#667085; margin:0 0 18px 0; font-size:.98rem; }}
  .hero {{ margin:0 0 18px 0; }}
  .hero h1 {{ font-size:1.7rem; margin:0 0 4px 0; color:#101828; }}
  .hero p {{ color:#667085; margin:0; font-size:.98rem; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF; border-radius:12px; border-color:#E4E7EC !important;
        box-shadow:0 1px 2px rgba(16,24,40,.04); }}
  div.stButton > button {{ border-radius:8px; border:1px solid #D0D5DD; font-weight:500; }}
  div.stButton > button[kind="primary"] {{ background:{TEAL}; border:none; color:#fff; padding:.5rem 1.2rem; }}
  div.stButton > button[kind="primary"]:disabled {{ background:#CBD2D9; color:#fff; opacity:1; }}
  button[data-baseweb="tab"] {{ font-weight:600; }}
  /* source selector: heading-style tabs */
  [data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {{ padding:.6rem 1.6rem; min-height:2.9rem; border:1.5px solid #D0D5DD; background:#fff; }}
  [data-testid="stSegmentedControl"] button *, [data-testid="stButtonGroup"] button * {{ font-size:1.05rem !important; font-weight:700 !important; color:#344054; }}
  [data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
  [data-testid="stButtonGroup"] button[kind="segmented_controlActive"], [data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:{TEAL} !important; border-color:{TEAL} !important; }}
  [data-testid="stSegmentedControl"] button[aria-checked="true"] *, [data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] *,
  [data-testid="stButtonGroup"] button[kind="segmented_controlActive"] *, [data-testid="stButtonGroup"] button[aria-checked="true"] * {{ color:#fff !important; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:#667085 !important; }}
  /* question box: highlighted input */
  .stTextArea textarea {{ background:#fff !important; border:1.5px solid {TEAL} !important; border-radius:10px; font-size:1.02rem; }}
  .stTextArea textarea:focus {{ box-shadow:0 0 0 3px rgba(15,118,110,.18) !important; }}
  .stTextArea [data-testid="stWidgetLabel"] p {{ font-size:1rem; font-weight:700; color:#101828; }}
  .sectionlabel {{ font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#667085; margin-bottom:4px; }}
  .answer {{ border-radius:10px; padding:16px 18px; border:1px solid; }}
  .answer .body {{ font-size:1.18rem; line-height:1.65; margin-top:10px; color:#0B1220; font-weight:500; }}
  .answer .quote {{ background:rgba(15,118,110,.12); border-bottom:2px solid rgba(15,118,110,.45); padding:0 3px; border-radius:3px; font-weight:600; }}
  .answer .cite {{ font-family:ui-monospace,Menlo,monospace; font-size:.74rem; font-weight:500; color:#475467; background:#fff;
                   border:1px solid #D0D5DD; border-radius:6px; padding:1px 6px; margin-left:2px; white-space:nowrap; }}
  [data-testid="stWidgetLabel"] p {{ font-size:.86rem; font-weight:600; color:#344054; }}
  [data-testid="stCaptionContainer"] {{ color:#667085; }}
  .question {{ font-size:.95rem; color:#667085; margin-bottom:12px; font-weight:400; }}
  .answer.ok {{ background:#F0FAF8; border-color:#B7E0D9; }}
  .answer.none {{ background:#F2F5FA; border-color:#CCD6E6; }}
  .answer.user {{ background:#F6F3FC; border-color:#D9CCF0; }}
  .answer.esc {{ background:#FFF7E8; border-color:#F0D9A8; }}
  .badge {{ display:inline-block; font-size:.74rem; font-weight:650; padding:2px 10px; border-radius:99px; color:#fff; }}
  .answer.ok .badge {{ background:{TEAL}; }}
  .answer.none .badge {{ background:#52668A; }}
  .answer.user .badge {{ background:#6B4BA8; }}
  .answer.esc .badge {{ background:{AMBER}; }}
  .pills {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 6px 0; }}
  .pill {{ font-size:.82rem; color:#344054; background:#F2F4F7; border:1px solid #E4E7EB; border-radius:99px; padding:3px 12px; }}
  .pill .k {{ color:#667085; margin-right:6px; }}
  .pill.good {{ background:#ECFDF3; border-color:#ABEFC6; color:#067647; }}
  .pill.warn {{ background:#FFFAEB; border-color:#FEDF89; color:#B54708; }}
  .kpi {{ background:#fff; border:1px solid #E4E7EC; border-radius:12px; padding:14px 18px; }}
  .kpi .v {{ font-size:1.8rem; font-weight:700; color:#101828; line-height:1.1; }}
  .kpi .l {{ font-size:.84rem; color:#667085; margin-top:4px; }}
  .kpi.good .v {{ color:{TEAL}; }}
  .kpi.bad .v {{ color:{AMBER}; }}
  .note {{ color:#667085; font-size:.88rem; }}
  .finding {{ background:#F0FAF8; border-left:4px solid {TEAL}; border-radius:8px; padding:10px 16px; margin:6px 0 14px 0;
              color:#101828; font-size:.98rem; }}
  .src {{ font-family:ui-monospace,Menlo,monospace; font-size:.8rem; background:#F9FAFB; border:1px solid #E4E7EC;
          border-radius:8px; padding:6px 10px; margin:4px 0; display:flex; justify-content:space-between; }}
  .chip {{ display:inline-block; font-size:.74rem; font-weight:700; padding:1px 10px; border-radius:99px; color:#fff; }}
  .chip.Low {{ background:{TEAL}; }} .chip.Moderate {{ background:#3B5B87; }}
  .chip.High {{ background:{AMBER}; }} .chip.Critical {{ background:#B42318; }} .chip.na {{ background:{GREY}; }}
  .strip {{ display:flex; gap:14px; margin-bottom:18px; }}
  .strip .kpi {{ flex:1; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
  .dot.ready {{ background:#12B76A; }} .dot.busy {{ background:#F79009; }} .dot.off {{ background:#98A2B3; }}
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
MAX_QUEUE = 10   # how many people may wait for the model before new requests are turned away
TIME = {"C0": "5 to 15 s", "C1": "5 to 15 s", "C2": "about 30 s", "C3": "5 to 15 s", "C4": "about 30 s",
        "C5": "1 to 2 min", "C6": "1 to 2 min"}
DESC = {"C0": "The model answers from the five most relevant label excerpts, with no additional safeguards. Used as the baseline.",
        "C1": "Every statement must carry an exact quote and citation from the label, otherwise the system declines to answer.",
        "C2": "A second model verifies each statement against the label and removes any that are unsupported.",
        "C3": "Declines to answer when the best-matching excerpt falls below a minimum relevance threshold.",
        "C4": "The model generates verification questions about its own draft and revises the draft accordingly.",
        "C5": "All safeguards applied in sequence: evidence threshold, source quoting, verification questions and claim verification.",
        "C6": "C5 plus risk-tiered review: high-risk questions (dosage, interactions) receive stricter checks and are referred for clinician review."}
EXAMPLES = {"Adverse reactions": ("warfarin", "What are the most common serious adverse reactions of warfarin?"),
            "Dosage": ("furosemide", "What is the usual starting dose of furosemide tablets for edema in adults?"),
            "Interactions": ("warfarin", "Can I take warfarin together with aspirin?"),
            "Outside labeling": ("amoxicillin", "What is the best brand of amoxicillin to buy online?")}


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
        if S["waiting"] >= MAX_QUEUE:
            raise RuntimeError("queue_full")
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


def serve_doc(question, udoc, status):
    """Answer from the visitor's own uploaded document. Uses the same shared model lock and queue."""
    from userdoc import answer_from_doc
    S = service()
    with S["mu"]:
        if S["waiting"] >= MAX_QUEUE:
            raise RuntimeError("queue_full")
        ahead = S["waiting"] + (1 if S["busy"] else 0)
        S["waiting"] += 1
    if ahead:
        status.update(label=f"Waiting for the model: {ahead} request(s) ahead of you")
    with S["lock"]:
        with S["mu"]:
            S["waiting"] -= 1
            S["busy"] = True
        status.update(label="Searching your document...")
        try:
            res = answer_from_doc(question, udoc, get_retriever().model, MODEL, K)
        finally:
            with S["mu"]:
                S["busy"] = False
    with S["mu"]:
        S["served"] += 1
    return res


def load_doc(uploaded):
    """Read, split and embed an uploaded file. Held only in this visitor's session memory."""
    from userdoc import extract_text, build_index
    S = service()
    text = extract_text(uploaded.name, uploaded.getvalue())
    with S["lock"]:   # embedding uses the CPU model, so take a turn like everyone else
        return build_index(get_retriever().model, uploaded.name, text)


def reuse(drug, q):
    st.session_state["mode"] = "FDA drug labeling"
    st.session_state["q"] = q
    if drug in drug_list():
        st.session_state["drug"] = drug


def use_example(drug, q):
    st.session_state["q"] = q
    if drug in drug_list():
        st.session_state["drug"] = drug


@st.cache_data(ttl=20, show_spinner=False)
def model_available():
    """True if a local Ollama server answers. On a hosted copy of the app (no model) this is False,
    and the Ask page switches itself off while the Results page keeps working."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5).read(1)
        return True
    except Exception:
        return False


def system_status():
    S = service()
    with S["mu"]:
        busy, waiting, served = S["busy"], S["waiting"], S["served"]
    return busy, waiting, served


def pills(items):
    """Small label/value chips shown under an answer."""
    out = ""
    for k, v, cls in items:
        out += f'<span class="pill {cls}"><span class="k">{html.escape(k)}</span>{v}</span>'
    return f'<div class="pills">{out}</div>'


def answer_card(ans, kind, badge):
    body = html.escape(ans).replace("\n", "<br>")
    body = re.sub(r"&quot;(.+?)&quot;", r'<span class="quote">\1</span>', body)          # exact quotes from the source
    body = re.sub(r"\[([A-Za-z0-9_\-]{6,})\]", r'<span class="cite">\1</span>', body)   # passage IDs
    return f'<div class="answer {kind}"><span class="badge">{badge}</span><div class="body">{body}</div></div>'


def load_uploaded(up):
    """Read an uploaded file once per file and keep it in this visitor's session only."""
    from userdoc import DocError
    if up is None:
        for k in ("udoc", "udoc_name"):
            st.session_state.pop(k, None)
        return
    if st.session_state.get("udoc_name") == (up.name, up.size):
        return
    try:
        with st.spinner("Reading your document..."):
            st.session_state["udoc"] = load_doc(up)
            st.session_state["udoc_name"] = (up.name, up.size)
    except DocError as e:
        st.session_state.pop("udoc", None)
        st.session_state.pop("udoc_name", None)
        st.warning(str(e))
    except Exception as e:
        st.session_state.pop("udoc", None)
        st.session_state.pop("udoc_name", None)
        st.error("That file could not be read.")
        st.exception(e)


def render_result(last):
    res, extra = last["res"], last["res"].get("extra") or {}
    ans, is_doc = res["answer"], last["kind"] == "doc"
    refused = "INSUFFICIENT EVIDENCE" in ans.upper()
    if is_doc:
        kind = "none" if refused else "user"
        badge = "No supported answer found in the uploaded document" if refused else "From uploaded document. Not verified against FDA labeling."
    elif "clinician review" in ans.lower():
        kind, badge = "esc", "Clinician review required"
    elif refused:
        kind, badge = "none", "No supported answer found in the drug labeling"
    else:
        kind, badge = "ok", "Answer"

    st.markdown('<div class="sectionlabel">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="question">{html.escape(last["question"])}</div>', unsafe_allow_html=True)
    st.markdown(answer_card(ans, kind, badge), unsafe_allow_html=True)

    import json as _json
    import streamlit.components.v1 as components
    cb1, cb2, _sp = st.columns([1.1, 1.3, 4])
    with cb2:
        cites = "\n".join(f"- {x['chunk_id']} (score {x['score']})" for x in res.get('sources', []))
        st.download_button("Download", f"Question: {last['question']}\n\nAnswer:\n{ans}\n\nCitations:\n{cites}\n", file_name="answer.txt",
                           key="dl_" + str(abs(hash(ans + last['question']))), use_container_width=True)
    with cb1:
      components.html(
        '<style>body{margin:0}button{font:500 13px -apple-system,Segoe UI,sans-serif;color:#344054;background:#fff;border:1px solid #D0D5DD;'
        'border-radius:8px;padding:0 14px;height:38px;width:100%;cursor:pointer}button:hover{background:#F2F4F7}</style>'
        '<button id="b">Copy answer</button><script>const t=' + _json.dumps(ans).replace("</", "<\\/") + ';'
        'document.getElementById("b").onclick=async function(){let ok=false;try{await navigator.clipboard.writeText(t);ok=true}catch(e){}'
        'if(!ok){const a=document.createElement("textarea");a.value=t;document.body.appendChild(a);a.select();'
        'try{ok=document.execCommand("copy")}catch(e){}a.remove()}this.textContent=ok?"Copied":"Copy failed";'
        'setTimeout(()=>this.textContent="Copy answer",1800)}</script>', height=40)

    items = []
    top = res.get("top_score")
    if not is_doc:
        tier = extra.get("severity_predicted")
        if tier in ("Low", "Moderate", "High", "Critical"):
            items.append(("Risk level", f'<span class="chip {tier}">{tier}</span>', ""))
        if top is not None:
            thr = 0.73 if tier in ("High", "Critical") else 0.71
            items.append(("Evidence strength", "Strong" if top >= thr else "Weak", "good" if top >= thr else "warn"))
    if top is not None:
        items.append(("Retrieval score", f"{top:.2f}", ""))
    items.append(("Response time", "reused" if last.get("cached") else f'{last["dt"]:.1f} s', ""))
    if not is_doc:
        items.append(("Verification", html.escape(last["cond"]), ""))
    else:
        items.append(("Document", html.escape(last["doc"]), ""))
    st.markdown(pills(items), unsafe_allow_html=True)

    has_steps = any(k in extra for k in ("draft", "checks", "revised", "verdicts"))
    names = ["Citations"] + (["Verification details"] if (has_steps and not is_doc) else [])
    tabs = st.tabs(names)
    with tabs[0]:
        for s_ in res.get("sources", []):
            st.markdown(f'<div class="src"><span>{html.escape(s_["chunk_id"])}</span><span>score {s_["score"]}</span></div>',
                        unsafe_allow_html=True)
            if s_.get("text"):
                with st.expander("View excerpt"):
                    st.write(s_["text"])
    if len(tabs) > 1:
        with tabs[1]:
            if "draft" in extra:
                st.markdown("**Initial draft**"); st.write(extra["draft"])
            if extra.get("checks"):
                st.markdown("**Verification questions**")
                for c in extra["checks"]:
                    st.write(f"- {c['question']}  →  {c['answer']}")
            if "revised" in extra:
                st.markdown("**Revised answer**"); st.write(extra["revised"])
            if extra.get("verdicts"):
                st.markdown("**Claim-level verification**")
                for v in extra["verdicts"]:
                    st.write(("✅ " if v["supported"] else "❌ ") + v["claim"])


# ----------------------------------------------------------------- page: Ask
def page_ask():
    live = model_available()
    st.markdown('<div class="pagehead"><h1>Ask a question</h1>'
                '<p>Answers grounded in FDA drug labeling or in a document you provide, with citations to the source excerpts.</p></div>',
                unsafe_allow_html=True)
    if not live:
        st.info("The answering service is currently unavailable. The Evaluation page remains available.")

    labels, mydoc = "FDA drug labeling", "Uploaded document"
    with st.container(border=True):
        mode = st.segmented_control("Source", [labels, mydoc], default=labels, key="mode", label_visibility="collapsed") or labels
        drug, up = None, None
        if mode == labels:
            drug = st.selectbox("Drug name", ["All drugs"] + drug_list(), key="drug")
            st.caption("Example queries")
            ex = st.columns(len(EXAMPLES))
            for col, (label, (d_, q_)) in zip(ex, EXAMPLES.items()):
                col.button(label, key="ex_" + label, on_click=use_example, args=(d_, q_), use_container_width=True)
        else:
            up = st.file_uploader("Upload document", type=["pdf", "docx", "txt"], key="udoc_file",
                                  help="PDF, Word or text file, up to 5 MB (about 50 pages). Kept in this session only.")
            load_uploaded(up)
            ud = st.session_state.get("udoc")
            if ud:
                st.caption(f"Ready: {ud['name']} ({len(ud['chunks'])} excerpt(s)). Retained for this session only.")

        question = st.text_area("Question", height=96, key="q",
                                placeholder="e.g. What is the usual starting dose of furosemide tablets for edema in adults?")
        cond = st.session_state.get("cond", "C1")
        if mode == labels:
            with st.expander("Advanced settings"):
                cond = st.selectbox("Verification level", list(VERSIONS), index=1, format_func=lambda x: VERSIONS[x], key="cond")
                st.caption(f"{DESC[cond]} Typical time: {TIME.get(cond, '-')}.")

        ready = bool(question.strip()) and live and (mode == labels or bool(st.session_state.get("udoc")))
        b1, b2 = st.columns([1, 3])
        go = b1.button("Submit", type="primary", disabled=not ready, use_container_width=True)
        busy, waiting, _ = system_status()
        b2.caption("For informational use only. Not medical advice. Do not enter patient information."
                   + (f" The service is busy; {waiting} request(s) queued." if busy and waiting else ""))

    asked = st.session_state.get("asked", 0)
    if go and asked >= MAX_PER_SESSION:
        st.warning("This session reached its limit of %d questions. Reload the page to continue." % MAX_PER_SESSION)
        go = False
    if go:
        try:
            with st.status("Starting...", expanded=False) as status:
                if mode == labels:
                    res, dt, cached = serve(cond, question.strip(), None if drug == "All drugs" else drug, status)
                    last = {"kind": "label", "res": res, "dt": dt, "cached": cached, "cond": cond, "question": question.strip()}
                else:
                    t0 = time.time()
                    res = serve_doc(question.strip(), st.session_state["udoc"], status)
                    last = {"kind": "doc", "res": res, "dt": time.time() - t0, "cached": False,
                            "question": question.strip(), "doc": st.session_state["udoc"]["name"]}
                status.update(label="Done", state="complete")
            st.session_state["asked"] = asked + 1
            st.session_state["last"] = last
            if mode == labels:
                h = [x for x in st.session_state.get("history", []) if x[1] != question.strip()]
                st.session_state["history"] = ([(drug, question.strip())] + h)[:5]
        except RuntimeError as e:
            if str(e) != "queue_full":
                raise
            st.warning("The service is at capacity (%d requests queued). Please try again shortly." % MAX_QUEUE)
        except Exception as e:
            st.error("The language model could not be reached. Confirm that the service is running and try again.")
            st.exception(e)

    last = st.session_state.get("last")
    if last and last["kind"] == ("label" if mode == labels else "doc"):
        with st.container(border=True):
            render_result(last)


# ----------------------------------------------------------------- page: Guide
def page_guide():
    st.markdown('<div class="pagehead"><h1>Guide</h1><p>How to use the assistant and how to interpret its responses.</p></div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Getting started")
        st.markdown("1. On **Ask**, select **FDA drug labeling** and choose a drug name, or leave it on All drugs.\n"
                    "2. Enter a question, or start from one of the example queries.\n"
                    "3. Select **Submit**. Responses typically take 5 to 30 seconds.\n"
                    "4. Review the answer, then open **Citations** to see the exact label excerpts it is based on.\n"
                    "5. Use **Copy answer** to copy the response text.")
    with st.container(border=True):
        st.markdown("#### Using an uploaded document")
        st.markdown("Select **Uploaded document**, upload a PDF, Word or text file (up to 5 MB, about 50 pages), and enter a question. "
                    "The document is retained for your session only. It is not stored and is not added to the FDA labeling. "
                    "Responses from an uploaded document are marked as not verified against FDA labeling. Scanned PDFs "
                    "without selectable text cannot be read.")
    with st.container(border=True):
        st.markdown("#### Interpreting a response")
        st.markdown("- **No supported answer found.** No supporting text was located in the source, so the system declined to answer.\n"
                    "- **Clinician review required.** The question concerns dosage or interactions and could not be confirmed with sufficient confidence.\n"
                    "- **Risk level.** The potential seriousness of an incorrect answer: Low, Moderate, High or Critical.\n"
                    "- **Evidence strength.** Whether the best-matching excerpt is a strong or weak match for the question.\n"
                    "- **Retrieval score.** Semantic similarity between the question and the closest excerpt, from 0 to 1.")
    with st.container(border=True):
        st.markdown("#### Verification levels (Advanced settings)")
        for c in VERSIONS:
            st.markdown(f"- **{VERSIONS[c]}** (typical response time {TIME[c]}): {DESC[c]}")
        st.caption("C1 is the default and is recommended for most questions.")
    with st.container(border=True):
        st.markdown("#### Limitations and good practice")
        st.markdown("- Coverage is limited to 20 FDA drug labels. Other drugs and sources are not included.\n"
                    "- Responses may still contain errors. Confirm clinically important information with a pharmacist or physician.\n"
                    "- For informational use only. Do not enter patient information.\n"
                    "- Each session is limited to %d questions. When demand is high, requests are queued." % MAX_PER_SESSION)
    with st.container(border=True):
        st.markdown("#### Running locally")
        st.markdown("Install the requirements, start the local model server with `ollama serve` (model `llama3.2:3b`), "
                    "then run `python -m streamlit run app/app.py`. The Evaluation page works without the model.")


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
    st.markdown("""<div class="hero"><h1>Evaluation</h1>
    <p>Eight versions of the system compared on the same 200 questions, graded by an AI judge that was checked against a human on 105 answers.
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
st.sidebar.markdown('<div class="brand"><span class="ico">💊</span>Drug Label Assistant</div><div class="brandsub">Answers grounded in FDA drug labeling</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Ask", "Evaluation", "Guide"], label_visibility="collapsed")
_b, _w, _n = system_status()
_state = ("off", "Offline") if not model_available() else (("busy", "Busy, %d waiting" % _w) if _b else ("ready", "Ready"))
st.sidebar.markdown("---")
st.sidebar.caption("Service status")
st.sidebar.markdown(f'<span class="dot {_state[0]}"></span>{_state[1]}', unsafe_allow_html=True)
_hist = st.session_state.get("history", [])
if _hist:
    st.sidebar.markdown("---")
    st.sidebar.caption("Recent questions")
    for _i, (_d, _q) in enumerate(_hist):
        st.sidebar.button(_q if len(_q) < 46 else _q[:43] + "...", key=f"hist{_i}", on_click=reuse, args=(_d, _q),
                          help=f"{_d or 'All drugs'}: {_q}", use_container_width=True)
st.sidebar.caption("%d quer%s answered" % (_n, "y" if _n == 1 else "ies"))
if page == "Ask":
    page_ask()
elif page == "Evaluation":
    page_results()
else:
    page_guide()