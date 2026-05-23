"""
╔══════════════════════════════════════════════════════════╗
║   PAU Student Performance Prediction App                 ║
║   Pan-Atlantic University · Lagos, Nigeria               ║
║   Version 2.0  |  2025/2026 Academic Session             ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import io
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG  (must be FIRST Streamlit call)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PAU · Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PAU Student Performance Prediction — Pan-Atlantic University, Lagos"},
)

# ═══════════════════════════════════════════════════════════
# BRAND PALETTE
# ═══════════════════════════════════════════════════════════
C = dict(
    navy="#0B1F4A", navy2="#162B5E", navy3="#1E3A7B",
    gold="#C9920A", gold_l="#F0B429", gold_p="#FDF3D8",
    white="#FFFFFF", off="#F7F8FC", border="#E2E6F0", muted="#6B7592",
    red="#B91C1C",   red_bg="#FEF2F2",   red_bd="#FECACA",
    green="#065F46", green_bg="#ECFDF5", green_bd="#A7F3D0",
    amber="#92400E", amber_bg="#FFFBEB", amber_bd="#FDE68A",
)

# ═══════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif !important; }}
.stApp {{ background: {C['off']} !important; }}
.block-container {{ padding-top: 1rem !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C['navy']} !important;
    border-right: 3px solid {C['gold']};
}}
[data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.88) !important; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {C['gold_l']} !important; font-size: 13px !important;
    letter-spacing: 1px; text-transform: uppercase;
}}
[data-testid="stSidebar"] label {{
    color: rgba(255,255,255,0.65) !important; font-size: 12px !important;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.10) !important; }}
[data-testid="stSidebar"] .stButton > button {{
    background: {C['gold']} !important; color: {C['navy']} !important;
    border: none !important; border-radius: 7px !important;
    font-weight: 700 !important; font-size: 13px !important;
    width: 100%; padding: 11px 0; letter-spacing: 0.3px;
}}
[data-testid="stSidebar"] .stButton > button:hover {{ background: {C['gold_l']} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 6px !important;
}}

/* ── Header ── */
.pau-header {{
    background: linear-gradient(135deg, {C['navy']} 0%, {C['navy2']} 60%, {C['navy3']} 100%);
    border-bottom: 4px solid {C['gold']};
    border-radius: 12px; padding: 20px 28px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 20px rgba(11,31,74,0.18);
}}
.pau-logo {{
    background: {C['gold']}; width: 56px; height: 56px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif; font-size: 20px;
    font-weight: 700; color: {C['navy']}; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(201,146,10,0.35);
}}
.pau-title h1 {{
    font-family: 'Playfair Display', serif !important; font-size: 23px !important;
    color: {C['white']} !important; margin: 0 !important; padding: 0 !important; line-height: 1.25;
}}
.pau-title p {{
    font-size: 11px; color: {C['gold_l']}; margin: 4px 0 0 0;
    letter-spacing: 1.8px; text-transform: uppercase;
}}
.pau-header-right {{ margin-left: auto; text-align: right; }}
.pau-badge {{
    display: inline-block; background: rgba(201,146,10,0.18);
    border: 1px solid {C['gold']}; color: {C['gold_l']};
    font-size: 11px; padding: 4px 12px; border-radius: 20px; letter-spacing: 0.5px;
}}

/* ── KPI cards ── */
.kpi-card {{
    background: {C['white']}; border: 1px solid {C['border']};
    border-top: 4px solid {C['gold']}; border-radius: 10px;
    padding: 18px 20px 14px; text-align: center;
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 4px 16px rgba(11,31,74,0.10); }}
.kpi-val {{ font-size: 32px; font-weight: 700; color: {C['navy']}; line-height: 1; }}
.kpi-lbl {{ font-size: 11px; color: {C['muted']}; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 5px; }}
.kpi-sub {{ font-size: 12px; margin-top: 5px; font-weight: 500; }}

/* ── Section headings ── */
.sec-head {{
    font-family: 'Playfair Display', serif; font-size: 16px; font-weight: 700;
    color: {C['navy']}; border-left: 4px solid {C['gold']};
    padding-left: 10px; margin: 22px 0 12px 0;
}}

/* ── Result cards ── */
.result-high {{ background:{C['red_bg']};border:1px solid {C['red_bd']};border-left:6px solid {C['red']};border-radius:10px;padding:18px 22px;color:{C['red']}; }}
.result-low  {{ background:{C['green_bg']};border:1px solid {C['green_bd']};border-left:6px solid {C['green']};border-radius:10px;padding:18px 22px;color:{C['green']}; }}
.result-med  {{ background:{C['amber_bg']};border:1px solid {C['amber_bd']};border-left:6px solid #D97706;border-radius:10px;padding:18px 22px;color:{C['amber']}; }}
.result-high h3, .result-low h3, .result-med h3 {{ margin:0 0 5px 0;font-size:18px;font-weight:700; }}
.result-high p,  .result-low p,  .result-med p  {{ margin:0;font-size:13px;opacity:0.9;line-height:1.6; }}

/* ── Insight box ── */
.insight-box {{
    background:{C['gold_p']};border:1px solid {C['gold']};
    border-radius:8px;padding:12px 16px;font-size:13px;
    color:{C['amber']};margin-top:10px;
}}

/* ── Probability bars ── */
.prob-row {{ margin-bottom: 8px; }}
.prob-label {{ font-size:12px;color:{C['muted']};display:flex;justify-content:space-between;margin-bottom:3px; }}
.prob-track {{ background:{C['border']};border-radius:4px;height:8px;overflow:hidden; }}
.prob-fill-risk {{ height:100%;border-radius:4px;background:linear-gradient(90deg,{C['red']},#EF4444); }}
.prob-fill-safe {{ height:100%;border-radius:4px;background:linear-gradient(90deg,{C['green']},#10B981); }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{C['navy']} !important;border-radius:10px 10px 0 0;padding:6px 10px 0;gap:4px;
}}
.stTabs [data-baseweb="tab"] {{
    color:rgba(255,255,255,0.60) !important;font-size:13px !important;
    padding:9px 20px !important;border-radius:7px 7px 0 0 !important;
}}
.stTabs [aria-selected="true"] {{
    background:{C['gold']} !important;color:{C['navy']} !important;font-weight:700 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background:{C['white']};border:1px solid {C['border']};border-top:none;
    border-radius:0 0 10px 10px;padding:24px;
}}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {{
    border:2px dashed {C['gold']} !important;border-radius:10px !important;
    background:{C['gold_p']} !important;padding:10px !important;
}}

/* ── Footer ── */
.pau-footer {{
    background:{C['navy']};color:rgba(255,255,255,0.38);text-align:center;
    font-size:11px;padding:16px;border-radius:10px;
    margin-top:32px;border-top:3px solid {C['gold']};letter-spacing:0.3px;
}}
.pau-footer b {{ color:{C['gold_l']}; }}
.gold-divider {{ height:2px;background:linear-gradient(90deg,{C['gold']},transparent);border:none;margin:18px 0; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

FEATURES = ["level", "attendance_percentage", "ca_score", "exam_score", "previous_gpa"]
FEAT_LABELS = {
    "level": "Academic Level",
    "attendance_percentage": "Attendance (%)",
    "ca_score": "CA Score",
    "exam_score": "Exam Score",
    "previous_gpa": "Previous GPA",
}

# ═══════════════════════════════════════════════════════════
# DATA & MODEL (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def build_model(_hash):
    df = pd.read_csv(DATA_PATH)
    clean = df.drop(columns=["student_id","name","department"], errors="ignore")
    X = clean[FEATURES];  y = clean["at_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:,1]
    cv      = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    m = dict(
        accuracy  = accuracy_score(y_te, y_pred),
        precision = precision_score(y_te, y_pred, zero_division=0),
        recall    = recall_score(y_te, y_pred, zero_division=0),
        f1        = f1_score(y_te, y_pred, zero_division=0),
        cv_mean   = cv.mean(), cv_std = cv.std(),
        report    = classification_report(y_te, y_pred, output_dict=True),
        cm        = confusion_matrix(y_te, y_pred),
        roc       = (fpr, tpr),
        roc_auc   = auc(fpr, tpr),
    )
    fi = pd.DataFrame({
        "Feature":    [FEAT_LABELS[f] for f in FEATURES],
        "Raw":        FEATURES,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    joblib.dump(clf, MODEL_PATH)
    return clf, m, fi

df_raw = load_data()
model, M, fi = build_model(str(len(df_raw)))
at_risk_n = int(df_raw["at_risk"].sum())
safe_n    = len(df_raw) - at_risk_n

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
def predict_single(level, att, ca, exam, gpa):
    row = pd.DataFrame([dict(level=level, attendance_percentage=att,
                             ca_score=ca, exam_score=exam, previous_gpa=gpa)])[FEATURES]
    pred  = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    return pred, float(proba[1]), float(proba[0])

def predict_batch(df_in):
    missing = [f for f in FEATURES if f not in df_in.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    out = df_in.copy()
    proba = model.predict_proba(out[FEATURES])
    out["predicted_risk"]     = model.predict(out[FEATURES])
    out["prob_at_risk_%"]     = (proba[:,1]*100).round(1)
    out["prob_not_at_risk_%"] = (proba[:,0]*100).round(1)
    out["risk_label"]         = out["predicted_risk"].map({1:"AT-RISK", 0:"SAFE"})
    return out, None

def risk_tier(p):
    if p >= 0.70: return "high", "HIGH RISK",     "⚠️"
    if p >= 0.40: return "med",  "MODERATE RISK",  "🔶"
    return          "low",  "LOW RISK",        "✅"

def tips_for(level, att, ca, exam, gpa):
    t = []
    if att < 60:   t.append("📅 <b>Attendance critically low.</b> Initiate attendance contract & weekly check-ins.")
    elif att < 75: t.append("📅 <b>Attendance needs improvement.</b> Send early-warning notice to advisor.")
    if ca < 40:    t.append("📝 <b>CA score very poor.</b> Enrol in tutoring programme and study groups immediately.")
    elif ca < 55:  t.append("📝 <b>CA below average.</b> Assign supplementary tasks; increase instructor contact hours.")
    if exam < 40:  t.append("📖 <b>Exam score critically low.</b> Provide exam-technique coaching & past-paper practice.")
    elif exam < 55:t.append("📖 <b>Exam score below average.</b> Encourage revision sessions with faculty support.")
    if gpa < 1.5:  t.append("🎓 <b>GPA dangerously low.</b> Trigger academic probation review and counselling referral.")
    elif gpa < 2.0:t.append("🎓 <b>GPA below passing threshold.</b> Schedule academic advisor meeting this week.")
    if level == 100:t.append("🏫 <b>First-year student.</b> Connect with peer mentor; ensure orientation support active.")
    if not t:      t.append("✅ <b>Student appears well-supported.</b> Maintain routine semester check-ins.")
    return t

def ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title,  color=C["navy"], fontweight="bold", fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=C["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)
    ax.tick_params(colors=C["muted"], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
    ax.set_facecolor(C["off"])

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="pau-header">
  <div class="pau-logo">PAU</div>
  <div class="pau-title">
    <h1>Student Performance Prediction System</h1>
    <p>Pan-Atlantic University &nbsp;·&nbsp; Predictive Analytics Platform &nbsp;·&nbsp; 2025/2026</p>
  </div>
  <div class="pau-header-right">
    <div class="pau-badge">🎓 Computer Science Department</div><br>
    <div class="pau-badge" style="margin-top:6px">🤖 Random Forest · {round(M['accuracy']*100,1)}% Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:12px 0 4px">
      <div style="display:inline-block;background:{C['gold']};padding:8px 22px;
           border-radius:8px;font-family:'Playfair Display',serif;
           font-size:15px;font-weight:700;color:{C['navy']};">PAU Analytics</div>
      <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:6px;
           letter-spacing:1px;">SINGLE STUDENT PREDICTION</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 Student Input")
    s_name  = st.text_input("Student Name (optional)", placeholder="e.g. Emeka Obi")
    s_level = st.selectbox("Academic Level", [100, 200, 300, 400])
    s_att   = st.slider("Attendance (%)", 0, 100, 75)
    s_ca    = st.slider("Continuous Assessment Score", 0, 100, 60)
    s_exam  = st.slider("Examination Score", 0, 100, 65)
    s_gpa   = st.number_input("Previous GPA (0.0 – 4.0)", 0.0, 4.0, 2.5, 0.01)
    st.markdown("---")
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:rgba(255,255,255,0.38);text-align:center;line-height:1.9">
      Algorithm: Random Forest (200 trees)<br>
      CV Accuracy: {round(M['cv_mean']*100,1)}% ± {round(M['cv_std']*100,1)}%<br>
      Training samples: {len(df_raw)} students<br>
      Features: 5 academic indicators
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════
k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    (k1, str(len(df_raw)),             "Total Students",   "",                                        C["navy"]),
    (k2, str(at_risk_n),               "At-Risk Students", f"{round(at_risk_n/len(df_raw)*100,1)}% of cohort", C["red"]),
    (k3, str(safe_n),                  "Not At-Risk",      f"{round(safe_n/len(df_raw)*100,1)}% of cohort",    C["green"]),
    (k4, f"{round(M['accuracy']*100,1)}%","Model Accuracy",f"CV: {round(M['cv_mean']*100,1)}%",      C["navy"]),
    (k5, f"{round(M['roc_auc'],3)}",   "ROC-AUC Score",   "Area Under Curve",                        C["navy"]),
]
for col, val, lbl, sub, color in kpis:
    sub_html = f'<div class="kpi-sub" style="color:{color}">{sub}</div>' if sub else ""
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-val" style="color:{color}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>{sub_html}
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PREDICTION RESULT
# ═══════════════════════════════════════════════════════════
if predict_btn:
    pred, p_risk, p_safe = predict_single(s_level, s_att, s_ca, s_exam, s_gpa)
    tier, tier_lbl, icon = risk_tier(p_risk)
    css = f"result-{'high' if tier=='high' else 'low' if tier=='low' else 'med'}"
    name_str = f"<b>{s_name}</b> — " if s_name.strip() else ""
    rec = tips_for(s_level, s_att, s_ca, s_exam, s_gpa)

    st.markdown(f"""
    <div class="{css}">
      <h3>{icon} {tier_lbl} — Prediction Result</h3>
      <p>{name_str}Level {s_level} student has a
      <b>{round(p_risk*100,1)}% probability</b> of being at academic risk.</p>
    </div>""", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec-head">Probability Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-label"><span>At-Risk Probability</span><span><b>{round(p_risk*100,1)}%</b></span></div>
          <div class="prob-track"><div class="prob-fill-risk" style="width:{round(p_risk*100,1)}%"></div></div>
        </div>
        <div class="prob-row" style="margin-top:10px">
          <div class="prob-label"><span>Safe Probability</span><span><b>{round(p_safe*100,1)}%</b></span></div>
          <div class="prob-track"><div class="prob-fill-safe" style="width:{round(p_safe*100,1)}%"></div></div>
        </div>""", unsafe_allow_html=True)

        # Student profile bar chart
        fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="none")
        norm_vals  = [s_level/4*100, s_att, s_ca, s_exam, s_gpa/4*100]
        feat_names = ["Level", "Attendance", "CA Score", "Exam Score", "GPA"]
        bar_cols   = [C["red"] if v < 50 else C["gold"] if v < 70 else C["navy"] for v in norm_vals]
        bars = ax.barh(feat_names, norm_vals, color=bar_cols, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, norm_vals):
            ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2,
                    f"{v:.0f}", va="center", fontsize=9, color=C["navy"], fontweight="600")
        ax.set_xlim(0, 115)
        ax.axvline(70, color=C["gold"], linestyle="--", lw=1, alpha=0.6, label="70 target")
        ax.axvline(50, color=C["red"],  linestyle="--", lw=1, alpha=0.5, label="50 warning")
        ax.legend(fontsize=8, loc="lower right")
        ax_style(ax, "Student Profile (normalised out of 100)", "Score")
        fig.tight_layout(); fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with r2:
        st.markdown('<div class="sec-head">Intervention Recommendations</div>', unsafe_allow_html=True)
        for tip in rec:
            st.markdown(f'<div class="insight-box">{tip}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Dashboard",
    "🔬  Model Performance",
    "📤  Batch Predict",
    "📋  Student Data",
    "📈  EDA",
])

# ─────────────────────── TAB 1: DASHBOARD ───────────────────────
with tab1:
    st.markdown('<div class="sec-head">Cohort Risk Overview</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:  # Donut
        fig, ax = plt.subplots(figsize=(4, 3.8), facecolor="none")
        wedges, texts, auts = ax.pie(
            [safe_n, at_risk_n], labels=["Not At-Risk","At-Risk"],
            colors=[C["navy"], C["gold"]], autopct="%1.1f%%", startangle=90,
            pctdistance=0.78,
            wedgeprops={"linewidth":2.5,"edgecolor":"white","width":0.55},
            textprops={"fontsize":10}
        )
        for a in auts: a.set_color("white"); a.set_fontweight("bold"); a.set_fontsize(9)
        ax.add_patch(plt.Circle((0,0), 0.35, color=C["off"]))
        ax.text(0, 0, f"{len(df_raw)}\nStudents", ha="center", va="center",
                fontsize=10, fontweight="bold", color=C["navy"])
        ax.set_title("Risk Distribution", color=C["navy"], fontweight="bold", pad=8)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with d2:  # By level
        fig, ax = plt.subplots(figsize=(4, 3.8), facecolor="none")
        lr = df_raw.groupby("level")["at_risk"].agg(["mean","count"]).reset_index()
        lr["pct"] = lr["mean"] * 100
        bcols = [C["gold"] if p >= 80 else C["navy"] for p in lr["pct"]]
        bars  = ax.bar([f"L{l}" for l in lr["level"]], lr["pct"],
                       color=bcols, edgecolor="white", linewidth=1.5, width=0.55)
        for bar, cnt, p in zip(bars, lr["count"], lr["pct"]):
            ax.text(bar.get_x()+bar.get_width()/2, p+1, f"{p:.0f}%",
                    ha="center", va="bottom", fontsize=9, color=C["navy"], fontweight="bold")
            ax.text(bar.get_x()+bar.get_width()/2, p/2, f"n={cnt}",
                    ha="center", va="center", fontsize=8, color="white", fontweight="600")
        ax.set_ylim(0, 115)
        ax_style(ax, "At-Risk Rate by Level", "", "% At-Risk")
        fig.tight_layout(); fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with d3:  # GPA vs Attendance scatter
        fig, ax = plt.subplots(figsize=(4, 3.8), facecolor="none")
        for val, color, marker, label in [(0,C["navy"],"o","Not At-Risk"),(1,C["gold"],"^","At-Risk")]:
            sub = df_raw[df_raw["at_risk"]==val]
            ax.scatter(sub["attendance_percentage"], sub["previous_gpa"],
                       color=color, alpha=0.65, s=28, label=label, marker=marker, zorder=3)
        ax.axhline(2.0, color=C["red"], lw=1, ls="--", alpha=0.55, label="GPA=2.0")
        ax.axvline(65,  color=C["red"], lw=1, ls=":",  alpha=0.55, label="Att=65%")
        ax.legend(fontsize=8, framealpha=0.8)
        ax_style(ax, "GPA vs Attendance", "Attendance (%)", "Previous GPA")
        fig.tight_layout(); fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Box plots
    st.markdown('<div class="sec-head">Score Distributions by Risk Group</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    for col, feat, lbl in [(b1,"attendance_percentage","Attendance (%)"),(b2,"ca_score","CA Score"),(b3,"exam_score","Exam Score")]:
        with col:
            fig, ax = plt.subplots(figsize=(3.8, 3.2), facecolor="none")
            data_g = [df_raw[df_raw["at_risk"]==v][feat].dropna().values for v in [0,1]]
            bp = ax.boxplot(data_g, patch_artist=True,
                            medianprops={"color":C["gold"],"linewidth":2},
                            whiskerprops={"color":C["muted"]}, capprops={"color":C["muted"]},
                            flierprops={"marker":"o","markersize":4,"markerfacecolor":C["muted"],"alpha":0.5})
            bp["boxes"][0].set_facecolor(C["navy"]+"AA")
            bp["boxes"][1].set_facecolor(C["gold"]+"AA")
            ax.set_xticks([1,2]); ax.set_xticklabels(["Not At-Risk","At-Risk"], fontsize=9)
            ax_style(ax, lbl, "", lbl)
            fig.tight_layout(); fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True); plt.close()

    # Feature importance
    st.markdown('<div class="sec-head">Feature Importance</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="none")
    fi_cols = [C["gold"] if i==0 else C["navy"] if i==1 else C["navy2"] for i in range(len(fi))]
    bars = ax.barh(fi["Feature"], fi["Importance"], color=fi_cols, edgecolor="white", linewidth=0.8, height=0.55)
    for bar in bars:
        w = bar.get_width()
        ax.text(w+0.002, bar.get_y()+bar.get_height()/2, f"{w:.3f}",
                va="center", fontsize=9, color=C["navy"], fontweight="600")
    ax.set_xlim(0, fi["Importance"].max()*1.22); ax.invert_yaxis()
    ax_style(ax, "What Drives At-Risk Predictions?", "Importance Score")
    fig.tight_layout(); fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True); plt.close()

# ─────────────────────── TAB 2: MODEL PERFORMANCE ───────────────
with tab2:
    st.markdown('<div class="sec-head">Performance Metrics</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    for col, key, lbl in [(m1,"accuracy","Accuracy"),(m2,"precision","Precision"),(m3,"recall","Recall"),(m4,"f1","F1-Score")]:
        v = round(M[key]*100, 1)
        clr = C["green"] if v>=85 else C["gold"] if v>=70 else C["red"]
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-val" style="color:{clr}">{v}%</div>
              <div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<div class="sec-head">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        sns.heatmap(M["cm"], annot=True, fmt="d",
                    cmap=sns.light_palette(C["navy"], as_cmap=True), ax=ax,
                    xticklabels=["Not At-Risk","At-Risk"],
                    yticklabels=["Not At-Risk","At-Risk"],
                    linewidths=1.5, linecolor="white",
                    annot_kws={"size":16,"weight":"bold"})
        ax.set_xlabel("Predicted", color=C["muted"])
        ax.set_ylabel("Actual", color=C["muted"])
        ax.set_title("Confusion Matrix", color=C["navy"], fontweight="bold", pad=10)
        ax.tick_params(colors=C["muted"])
        fig.patch.set_alpha(0); fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with p2:
        st.markdown('<div class="sec-head">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr = M["roc"]
        fig, ax = plt.subplots(figsize=(5,4), facecolor="none")
        ax.plot(fpr, tpr, color=C["gold"], lw=2.5, label=f"AUC = {M['roc_auc']:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color=C["gold"])
        ax.plot([0,1],[0,1], color=C["border"], lw=1.5, ls="--", label="Random")
        ax.legend(fontsize=10, framealpha=0.9)
        ax_style(ax, "ROC Curve — Discrimination Ability", "False Positive Rate", "True Positive Rate")
        fig.patch.set_alpha(0); fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="sec-head">Full Classification Report</div>', unsafe_allow_html=True)
    rpt = M["report"]
    report_df = pd.DataFrame({
        "Class":     ["Not At-Risk (0)","At-Risk (1)","Macro Avg","Weighted Avg"],
        "Precision": [rpt["0"]["precision"],rpt["1"]["precision"],rpt["macro avg"]["precision"],rpt["weighted avg"]["precision"]],
        "Recall":    [rpt["0"]["recall"],   rpt["1"]["recall"],   rpt["macro avg"]["recall"],   rpt["weighted avg"]["recall"]],
        "F1-Score":  [rpt["0"]["f1-score"], rpt["1"]["f1-score"], rpt["macro avg"]["f1-score"], rpt["weighted avg"]["f1-score"]],
        "Support":   [int(rpt["0"]["support"]),int(rpt["1"]["support"]),int(rpt["macro avg"]["support"]),int(rpt["weighted avg"]["support"])],
    }).set_index("Class").round(4)
    st.dataframe(report_df.style.format("{:.4f}", subset=["Precision","Recall","F1-Score"]),
                 use_container_width=True)

    st.markdown(f"""<div class="insight-box">
      💡 <b>Cross-Validation (5-fold):</b> Mean Accuracy = <b>{round(M['cv_mean']*100,1)}%</b>
      &nbsp;±&nbsp; <b>{round(M['cv_std']*100,2)}%</b> — stable, well-generalising model.
    </div>""", unsafe_allow_html=True)

# ─────────────────────── TAB 3: BATCH PREDICT ───────────────────
with tab3:
    st.markdown('<div class="sec-head">Batch Student Prediction via CSV Upload</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="insight-box">
      📋 Upload a CSV with student records. Required columns:
      <b>level, attendance_percentage, ca_score, exam_score, previous_gpa</b>.
      Optional columns (name, student_id, etc.) will be preserved.
    </div>""", unsafe_allow_html=True)

    template_df = pd.DataFrame({
        "student_id":[2001,2002,2003],"name":["Student A","Student B","Student C"],
        "level":[100,200,300],"attendance_percentage":[75,60,88],
        "ca_score":[65,45,78],"exam_score":[70,50,82],"previous_gpa":[3.1,1.8,3.5],
    })
    st.download_button("⬇️  Download Template CSV",
                       data=template_df.to_csv(index=False).encode(),
                       file_name="pau_prediction_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅  Loaded {len(batch_df)} student records.")
            result_df, err = predict_batch(batch_df)
            if err:
                st.error(f"❌  {err}")
            else:
                b_risk = (result_df["predicted_risk"]==1).sum()
                b_safe = (result_df["predicted_risk"]==0).sum()
                ba, bb, bc = st.columns(3)
                with ba: st.markdown(f"""<div class="kpi-card"><div class="kpi-val">{len(result_df)}</div><div class="kpi-lbl">Students Processed</div></div>""", unsafe_allow_html=True)
                with bb: st.markdown(f"""<div class="kpi-card"><div class="kpi-val" style="color:{C['red']}">{b_risk}</div><div class="kpi-lbl">Predicted At-Risk</div></div>""", unsafe_allow_html=True)
                with bc: st.markdown(f"""<div class="kpi-card"><div class="kpi-val" style="color:{C['green']}">{b_safe}</div><div class="kpi-lbl">Predicted Safe</div></div>""", unsafe_allow_html=True)

                st.markdown('<div class="sec-head">Prediction Results</div>', unsafe_allow_html=True)
                def hl_batch_row(row):
                    if row.get("risk_label") == "AT-RISK":
                        return [f"background-color:#FEF2F2;color:{C['red']};font-weight:700"
                                if col == "risk_label" else "background-color:#FEF2F2"
                                for col in result_df.columns]
                    return [f"background-color:#ECFDF5;color:{C['green']};font-weight:700"
                            if col == "risk_label" else "background-color:#ECFDF5"
                            for col in result_df.columns]
                st.dataframe(result_df.style.apply(hl_batch_row, axis=1),
                             use_container_width=True, height=380)
                st.download_button("⬇️  Download Predictions CSV",
                                   data=result_df.to_csv(index=False).encode(),
                                   file_name="pau_batch_predictions.csv", mime="text/csv")
        except Exception as e:
            import traceback
            st.error(f"❌  Error reading file: {e}")
            st.code(traceback.format_exc(), language="text")
    else:
        st.info("👆  Upload a CSV file above to generate batch predictions.")

# ─────────────────────── TAB 4: STUDENT DATA ────────────────────
with tab4:
    st.markdown('<div class="sec-head">Full Student Dataset</div>', unsafe_allow_html=True)
    fa, fb, fc = st.columns([1.2, 1.5, 1.5])
    with fa:
        risk_filter  = st.selectbox("Filter by Risk", ["All Students","At-Risk Only","Not At-Risk Only"])
    with fb:
        level_filter = st.multiselect("Filter by Level", [100,200,300,400], default=[100,200,300,400])
    with fc:
        search = st.text_input("Search by student name", placeholder="Type a name...")

    filtered = df_raw.copy()
    if risk_filter  == "At-Risk Only":       filtered = filtered[filtered["at_risk"]==1]
    elif risk_filter == "Not At-Risk Only":  filtered = filtered[filtered["at_risk"]==0]
    if level_filter: filtered = filtered[filtered["level"].isin(level_filter)]
    if search.strip(): filtered = filtered[filtered["name"].str.contains(search.strip(), case=False, na=False)]

    def hl_row(row):
        bg = "#FEF2F2" if row["at_risk"]==1 else "#ECFDF5"
        return [f"background-color:{bg}"] * len(row)

    st.dataframe(filtered.style.apply(hl_row, axis=1), use_container_width=True, height=440)
    ci1, ci2 = st.columns([2,1])
    with ci1: st.caption(f"Showing **{len(filtered)}** of **{len(df_raw)}** records")
    with ci2:
        st.download_button("⬇️  Export Filtered Data",
                           data=filtered.to_csv(index=False).encode(),
                           file_name="pau_filtered_students.csv", mime="text/csv")

# ─────────────────────── TAB 5: EDA ─────────────────────────────
with tab5:
    numeric_cols = ["attendance_percentage","ca_score","exam_score","previous_gpa"]

    st.markdown('<div class="sec-head">Feature Distributions (At-Risk vs Not At-Risk)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(14,4), facecolor="none")
    for ax, col in zip(axes, numeric_cols):
        for val, clr in [(0,C["navy"]),(1,C["gold"])]:
            ax.hist(df_raw[df_raw["at_risk"]==val][col].dropna(),
                    bins=16, alpha=0.72, color=clr, edgecolor="white",
                    label=("Not At-Risk" if val==0 else "At-Risk"))
        ax_style(ax, col.replace("_"," ").title(), col.replace("_"," ").title(), "Count")
    fig.legend(handles=[mpatches.Patch(color=C["navy"],label="Not At-Risk"),
                        mpatches.Patch(color=C["gold"],label="At-Risk")],
               loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5,1.04))
    fig.tight_layout(); fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True); plt.close()

    e1, e2 = st.columns(2)

    with e1:
        st.markdown('<div class="sec-head">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5), facecolor="none")
        sns.heatmap(df_raw[numeric_cols+["at_risk"]].corr(), annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(220,40,as_cmap=True), ax=ax,
                    linewidths=0.6, linecolor="white", annot_kws={"size":10,"weight":"bold"})
        ax.set_title("Feature Correlation Matrix", color=C["navy"], fontweight="bold", pad=10)
        ax.tick_params(colors=C["muted"], labelsize=9)
        fig.patch.set_alpha(0); fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with e2:
        st.markdown('<div class="sec-head">CA Score vs Exam Score</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5), facecolor="none")
        for val, clr, mk, lbl in [(0,C["navy"],"o","Not At-Risk"),(1,C["gold"],"^","At-Risk")]:
            sub = df_raw[df_raw["at_risk"]==val]
            ax.scatter(sub["ca_score"], sub["exam_score"], color=clr, alpha=0.65, s=32,
                       label=lbl, marker=mk, zorder=3)
            z = np.polyfit(sub["ca_score"], sub["exam_score"], 1)
            xs = np.linspace(sub["ca_score"].min(), sub["ca_score"].max(), 80)
            ax.plot(xs, np.poly1d(z)(xs), color=clr, lw=1.5, ls="--", alpha=0.55)
        ax.legend(fontsize=10, framealpha=0.9)
        ax_style(ax, "CA Score vs Exam Score by Risk Group", "CA Score", "Exam Score")
        fig.patch.set_alpha(0); fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Violin plot
    st.markdown('<div class="sec-head">GPA Distribution by Level & Risk</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,4), facecolor="none")
    offset = 0; xticks = []; xlabels = []
    for lv in [100,200,300,400]:
        sub = df_raw[df_raw["level"]==lv]
        if len(sub) < 3: continue
        for i, (risk_val, clr) in enumerate([(0,C["navy"]),(1,C["gold"])]):
            data = sub[sub["at_risk"]==risk_val]["previous_gpa"].dropna().values
            if len(data) > 1:
                vp = ax.violinplot([data], positions=[offset+i*0.55], widths=0.45, showmedians=True)
                for pc in vp["bodies"]: pc.set_facecolor(clr); pc.set_alpha(0.65)
                vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(1.5)
        xticks.append(offset+0.275); xlabels.append(f"Level {lv}"); offset += 1.4
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.legend(handles=[mpatches.Patch(color=C["navy"],alpha=0.75,label="Not At-Risk"),
                       mpatches.Patch(color=C["gold"],alpha=0.75,label="At-Risk")], fontsize=10)
    ax_style(ax, "GPA Distribution by Level and Risk Group", "Academic Level", "Previous GPA")
    fig.patch.set_alpha(0); fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="pau-footer">
  © 2026 Pan-Atlantic University &nbsp;·&nbsp; Lagos, Nigeria
  &nbsp;|&nbsp; <b>Unlocking Potential · Building Responsible Leaders</b>
  &nbsp;|&nbsp; Predictive Analytics Platform v2.0
  &nbsp;|&nbsp; Computer Science Department
</div>
""", unsafe_allow_html=True)
