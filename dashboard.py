import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
# Enrolled dihapus sesuai notebook (df = df[df['Status'] != 'Enrolled'].copy())
COLORS       = {"Graduate": "#4CAF50", "Dropout": "#F44336"}
STATUS_ORDER = ["Graduate", "Dropout"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9f9f9",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Exact feature order used during notebook training ────────────────────────
FEATURE_COLS = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification",
    "Previous_qualification_grade", "Nacionality",
    "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation",
    "Admission_grade", "Displaced", "Educational_special_needs",
    "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
    "Age_at_enrollment", "International",
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]

# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset/data.csv", sep=";")


@st.cache_resource
def train_model():
    """
    Mirrors notebook preprocessing exactly:
    - Drop 'Enrolled' rows (binary classification: Graduate vs Dropout)
    - LabelEncoder on Status  (Dropout→0, Graduate→1)
    - X = df.drop('Status'), column order preserved from dataset
    - 80/20 stratified split, random_state=42
    - StandardScaler fit on train only (applied only to Logistic Regression)
    - Train 4 models, pick best by Test Accuracy
    """
    df_raw = load_data()

    # ── Sesuai notebook: drop kelas 'Enrolled' ────────────────────────────────
    df = df_raw[df_raw["Status"] != "Enrolled"].copy()

    le = LabelEncoder()
    y  = le.fit_transform(df["Status"])   # Dropout=0, Graduate=1
    X  = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # ── 4 model sesuai notebook ───────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in models.items():
        Xtr = X_train_sc if name == "Logistic Regression" else X_train
        Xte = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy").mean()
        results.append({"Model": name, "Test Accuracy": acc, "CV Accuracy (5-fold)": cv})

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)

    # ── Best model ────────────────────────────────────────────────────────────
    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    Xte_best   = X_test_sc if best_name == "Logistic Regression" else X_test
    y_pred_best = best_model.predict(Xte_best)

    acc    = accuracy_score(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best,
                                   target_names=le.classes_, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred_best)

    return best_model, best_name, scaler, le, acc, report, cm, results_df, df


df_full                                                   = load_data()
best_model, best_name, scaler, le, acc, report, cm, results_df, df = train_model()

# Medians dihitung dari dataset SETELAH drop Enrolled (sesuai training data)
MEDIANS = df[FEATURE_COLS].median().to_dict()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Dashboard")
    st.markdown("---")
    page = st.radio("Menu", [
        "📊 Analysis",
        "🤖 Model Comparison",
        "🔮 Prediction",
        "📝 Recommendations",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"📁 {len(df):,} students (Graduate & Dropout)")
    st.caption(f"🤖 Best: **{best_name}** · Acc: **{acc:.2%}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Analysis":
    st.title("📊 Student Performance Analysis")
    st.info("ℹ️ Analisis dilakukan pada data **Graduate & Dropout** saja — kelas *Enrolled* dieksklusikan sesuai preprocessing notebook.", icon="ℹ️")

    total    = len(df)
    graduate = (df["Status"] == "Graduate").sum()
    dropout  = (df["Status"] == "Dropout").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Students",  f"{total:,}")
    c2.metric("🎓 Graduate",  f"{graduate:,}", f"{graduate/total*100:.1f}%")
    c3.metric("⚠️ Dropout",   f"{dropout:,}",  f"{dropout/total*100:.1f}%", delta_color="inverse")
    st.markdown("---")

    # ── Row 1: VIZ 1 & 2 ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1 · Status Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Status"].value_counts().reindex(STATUS_ORDER)
        _, _, autotexts = ax.pie(
            counts, labels=counts.index,
            colors=[COLORS[s] for s in counts.index],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
            textprops={"fontsize": 11},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title("Proporsi Status Mahasiswa", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### 2 · Semester 1 → 2 Grade Trajectory")
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in STATUS_ORDER:
            sub = df[df["Status"] == s]
            ax.scatter(sub["Curricular_units_1st_sem_grade"],
                       sub["Curricular_units_2nd_sem_grade"],
                       c=COLORS[s], label=s, alpha=0.35, s=15, edgecolors="none")
        lim = max(df["Curricular_units_1st_sem_grade"].max(),
                  df["Curricular_units_2nd_sem_grade"].max()) + 0.5
        ax.plot([0, lim], [0, lim], color="#bbb", linestyle="--", linewidth=1)
        ax.set_xlabel("Sem 1 Grade", fontsize=10)
        ax.set_ylabel("Sem 2 Grade", fontsize=10)
        ax.set_title("Nilai Sem 1 vs Sem 2 by Status", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, markerscale=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.caption("💡 Dropout cluster di (0,0) — gagal di dua semester sekaligus adalah sinyal paling kuat.")
    st.markdown("---")

    # ── Row 2: VIZ 3 & 4 ─────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 3 · Financial Factors vs Outcome")
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        for ax, col_name, lmap, xlabel in [
            (axes[0], "Tuition_fees_up_to_date", {0: "Overdue", 1: "Paid"},  "Tuition"),
            (axes[1], "Scholarship_holder",       {0: "No",      1: "Yes"},   "Scholarship"),
        ]:
            tmp = df.copy(); tmp[col_name] = tmp[col_name].map(lmap)
            ct  = pd.crosstab(tmp[col_name], tmp["Status"])[STATUS_ORDER]
            pct = ct.div(ct.sum(axis=1), axis=0) * 100
            bot = np.zeros(len(pct))
            for s in STATUS_ORDER:
                vals = pct[s].values
                bars = ax.bar(pct.index, vals, bottom=bot,
                              color=COLORS[s], label=s, edgecolor="white", width=0.5)
                for b, v, bv in zip(bars, vals, bot):
                    if v > 8:
                        ax.text(b.get_x()+b.get_width()/2, bv+v/2, f"{v:.0f}%",
                                ha="center", va="center", fontsize=8.5,
                                color="white", fontweight="bold")
                bot += vals
            ax.set_ylim(0, 110); ax.set_title(xlabel, fontsize=10, fontweight="bold")
            ax.set_ylabel("%" if ax == axes[0] else ""); ax.legend(fontsize=7, loc="upper right")
        fig.suptitle("Tuition & Scholarship vs Status (%)", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown("#### 4 · Courses Passed by Status")
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        for ax, col_name, title in [
            (axes[0], "Curricular_units_1st_sem_approved", "Sem 1"),
            (axes[1], "Curricular_units_2nd_sem_approved", "Sem 2"),
        ]:
            data_box = [df[df["Status"] == s][col_name].values for s in STATUS_ORDER]
            bp = ax.boxplot(data_box, patch_artist=True, labels=STATUS_ORDER,
                            widths=0.5, medianprops={"color": "white", "linewidth": 2})
            for patch, s in zip(bp["boxes"], STATUS_ORDER):
                patch.set_facecolor(COLORS[s]); patch.set_alpha(0.82)
            for w in bp["whiskers"]: w.set(color="#777", linewidth=1, linestyle="--")
            for cap in bp["caps"]:   cap.set(color="#777", linewidth=1.2)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticklabels(STATUS_ORDER, fontsize=7, rotation=12)
            ax.set_ylabel("Courses Passed" if ax == axes[0] else "")
        fig.suptitle("MK Lulus Sem 1 & 2 by Status", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.caption("💡 Graduate konsisten lulus lebih banyak MK — rendahnya MK lulus Sem 1 adalah early warning yang harus segera ditindak.")
    st.markdown("---")

    # ── Row 3: VIZ 5 & 6 ─────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### 5 · Enrollment Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in STATUS_ORDER:
            sub = df[df["Status"] == s]["Age_at_enrollment"]
            ax.hist(sub, bins=28, alpha=0.55, color=COLORS[s],
                    label=f"{s} (μ={sub.mean():.1f})", density=True)
        ax.axvline(23, color="#555", linestyle="--", linewidth=1.2)
        ax.text(23.5, ax.get_ylim()[1]*0.92, "Age 23", fontsize=8, color="#555")
        ax.set_xlabel("Age at Enrollment", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Distribusi Usia Pendaftaran", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col6:
        st.markdown("#### 6 · Debt & Gender Risk Profile")
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        for ax, col_name, lmap, xlabel in [
            (axes[0], "Debtor", {0: "No Debt", 1: "In Debt"}, "Debt Status"),
            (axes[1], "Gender", {0: "Female",  1: "Male"},     "Gender"),
        ]:
            tmp = df.copy(); tmp[col_name] = tmp[col_name].map(lmap)
            ct  = pd.crosstab(tmp[col_name], tmp["Status"])[STATUS_ORDER]
            pct = ct.div(ct.sum(axis=1), axis=0) * 100
            bot = np.zeros(len(pct))
            for s in STATUS_ORDER:
                vals = pct[s].values
                bars = ax.bar(pct.index, vals, bottom=bot,
                              color=COLORS[s], label=s, edgecolor="white", width=0.45)
                for b, v, bv in zip(bars, vals, bot):
                    if v > 8:
                        ax.text(b.get_x()+b.get_width()/2, bv+v/2, f"{v:.0f}%",
                                ha="center", va="center", fontsize=8,
                                color="white", fontweight="bold")
                bot += vals
            ax.set_ylim(0, 110); ax.set_title(xlabel, fontsize=10, fontweight="bold")
            ax.set_ylabel("%" if ax == axes[0] else ""); ax.legend(fontsize=7, loc="upper right")
        fig.suptitle("Debt & Gender vs Outcome (%)", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.caption("💡 Mahasiswa bermasalah finansial (debt + SPP nunggak) mendominasi grup dropout — intervensi keuangan adalah prioritas utama.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON (BARU: sesuai notebook Cell 35 & 36)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("Perbandingan 4 model sesuai notebook: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.")
    st.markdown("---")

    # ── Tabel hasil ──────────────────────────────────────────────────────────
    st.subheader("📋 Hasil Perbandingan Model")
    display_df = results_df.copy()
    display_df["Test Accuracy"]       = display_df["Test Accuracy"].map("{:.4f}".format)
    display_df["CV Accuracy (5-fold)"] = display_df["CV Accuracy (5-fold)"].map("{:.4f}".format)
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.success(f"🏆 Best Model: **{best_name}** — Test Accuracy **{acc:.2%}**")
    st.markdown("---")

    # ── Bar chart perbandingan ────────────────────────────────────────────────
    st.subheader("📊 Visualisasi Perbandingan Akurasi")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results_df))
    width = 0.35

    b1 = ax.bar(x - width/2, results_df["Test Accuracy"].astype(float),        width,
                label="Test Accuracy",        color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + width/2, results_df["CV Accuracy (5-fold)"].astype(float), width,
                label="CV Accuracy (5-fold)", color="#FF9800", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    ax.set_title("Perbandingan Performa Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    for bars in [b1, b2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Confusion Matrix & Classification Report best model ───────────────────
    st.subheader(f"📉 Detail Best Model: {best_name}")
    col1, col2 = st.columns(2)

    with col1:
        rows = []
        for cls in le.classes_:
            r = report[cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r["support"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col2:
        labels = list(le.classes_)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {best_name}", fontsize=11, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Student Status Prediction")
    st.markdown(f"Model: **{best_name}** (best model dari notebook) — Binary: **Graduate** vs **Dropout**.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    prev_grade      = c1.number_input("Previous Qualification Grade", 0.0, 200.0,
                                       float(MEDIANS["Previous_qualification_grade"]), 0.5)
    admission_grade = c2.number_input("Admission Grade", 0.0, 200.0,
                                       float(MEDIANS["Admission_grade"]), 0.5)
    age             = c3.number_input("Age at Enrollment", 17, 70,
                                       int(MEDIANS["Age_at_enrollment"]))

    c1, c2, c3, c4 = st.columns(4)
    s1_enrolled = c1.number_input("Sem 1 Enrolled",    0, 30, int(MEDIANS["Curricular_units_1st_sem_enrolled"]))
    s1_evals    = c2.number_input("Sem 1 Evaluations", 0, 30, int(MEDIANS["Curricular_units_1st_sem_evaluations"]))
    s1_approved = c3.number_input("Sem 1 Passed",      0, 30, int(MEDIANS["Curricular_units_1st_sem_approved"]))
    s1_grade    = c4.number_input("Sem 1 Grade",       0.0, 20.0, float(round(MEDIANS["Curricular_units_1st_sem_grade"], 1)), 0.1)

    c1, c2, c3, c4 = st.columns(4)
    s2_enrolled = c1.number_input("Sem 2 Enrolled",    0, 30, int(MEDIANS["Curricular_units_2nd_sem_enrolled"]))
    s2_evals    = c2.number_input("Sem 2 Evaluations", 0, 30, int(MEDIANS["Curricular_units_2nd_sem_evaluations"]))
    s2_approved = c3.number_input("Sem 2 Passed",      0, 30, int(MEDIANS["Curricular_units_2nd_sem_approved"]))
    s2_grade    = c4.number_input("Sem 2 Grade",       0.0, 20.0, float(round(MEDIANS["Curricular_units_2nd_sem_grade"], 1)), 0.1)

    c1, c2, c3, c4 = st.columns(4)
    scholarship   = c1.selectbox("Scholarship",  ["No", "Yes"])
    tuition_ok    = c2.selectbox("Tuition Paid", ["Yes", "No"])
    debtor        = c3.selectbox("Debtor",       ["No", "Yes"])
    gender        = c4.selectbox("Gender",       ["Female", "Male"])

    c1, c2, c3 = st.columns(3)
    displaced     = c1.selectbox("Displaced",     ["No", "Yes"])
    daytime       = c2.selectbox("Attendance",    ["Daytime", "Evening"])
    international = c3.selectbox("International", ["No", "Yes"])

    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        row = MEDIANS.copy()
        row.update({
            "Previous_qualification_grade":              prev_grade,
            "Admission_grade":                           admission_grade,
            "Age_at_enrollment":                         age,
            "Curricular_units_1st_sem_enrolled":         s1_enrolled,
            "Curricular_units_1st_sem_evaluations":      s1_evals,
            "Curricular_units_1st_sem_approved":         s1_approved,
            "Curricular_units_1st_sem_grade":            s1_grade,
            "Curricular_units_2nd_sem_enrolled":         s2_enrolled,
            "Curricular_units_2nd_sem_evaluations":      s2_evals,
            "Curricular_units_2nd_sem_approved":         s2_approved,
            "Curricular_units_2nd_sem_grade":            s2_grade,
            "Scholarship_holder":         1 if scholarship   == "Yes"     else 0,
            "Tuition_fees_up_to_date":    1 if tuition_ok    == "Yes"     else 0,
            "Debtor":                     1 if debtor        == "Yes"     else 0,
            "Gender":                     1 if gender        == "Male"    else 0,
            "Displaced":                  1 if displaced     == "Yes"     else 0,
            "Daytime_evening_attendance": 1 if daytime       == "Daytime" else 0,
            "International":              1 if international == "Yes"     else 0,
        })

        X_input = pd.DataFrame([row])[FEATURE_COLS]

        # Scaling hanya untuk Logistic Regression (sesuai notebook)
        if best_name == "Logistic Regression":
            X_proc = scaler.transform(X_input)
        else:
            X_proc = X_input.values

        proba      = best_model.predict_proba(X_proc)[0]
        pred_label = le.classes_[np.argmax(proba)]
        pred_prob  = proba[np.argmax(proba)]
        color = COLORS[pred_label]
        icon  = {"Graduate": "🎓", "Dropout": "⚠️"}[pred_label]

        st.markdown(
            f"<div style='background:{color}18;border-left:6px solid {color};"
            f"padding:18px 22px;border-radius:10px;margin:16px 0'>"
            f"<h2 style='color:{color};margin:0'>{icon} {pred_label}</h2>"
            f"<p style='margin:6px 0 0'>Confidence: "
            f"<strong style='color:{color}'>{pred_prob*100:.1f}%</strong></p>"
            f"</div>", unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(le.classes_, proba * 100,
                      color=[COLORS[c] for c in le.classes_],
                      edgecolor="white", alpha=0.9, width=0.4)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Class Probabilities", fontsize=12, fontweight="bold")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{p*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if pred_label == "Dropout":
            st.error("⚠️ Risiko dropout tinggi. Rekomendasikan konseling akademik segera dan review kondisi finansial mahasiswa.")
        else:
            st.success("🎓 Mahasiswa on track untuk lulus. Pertahankan dukungan yang ada!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Recommendations":
    st.title("📝 Kesimpulan & Rekomendasi")

    st.success(f"🏆 Best Model: **{best_name}** — Test Accuracy **{acc:.2%}** (Binary: Graduate vs Dropout)")

    # ── Model report ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")
    col1, col2 = st.columns(2)

    with col1:
        rows = []
        for cls in le.classes_:
            r = report[cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r["support"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col2:
        labels = list(le.classes_)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── Key findings ──────────────────────────────────────────────────────────
    st.subheader("🔍 Temuan Utama")
    findings = [
        ("#4CAF50", "🏆", "Performa akademik Sem 1 & 2 (nilai + MK lulus) adalah prediktor dropout terkuat — penurunan di Sem 1 sudah cukup jadi alarm."),
        ("#F44336", "💰", "Mahasiswa dengan SPP tunggak memiliki dropout rate ~3× lebih tinggi dibanding yang lunas — hambatan finansial mendominasi."),
        ("#2196F3", "🎓", "Penerima beasiswa lulus lebih tinggi secara signifikan — bantuan finansial langsung menekan angka dropout."),
        ("#FF9800", "📅", "Mahasiswa yang masuk di usia >23 tahun lebih rentan dropout, kemungkinan karena tanggung jawab lain di luar kampus."),
        ("#9C27B0", "💳", "Mahasiswa berstatus debtor terkonsentrasi di grup dropout — tekanan utang memperparah tantangan akademik."),
        ("#E91E63", "🌍", "Kondisi makroekonomi (GDP, unemployment) berpengaruh kecil namun terukur — relevan untuk kohort yang masuk di masa resesi."),
    ]
    for color, icon, text in findings:
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:10px 16px;"
            f"margin-bottom:8px;border-radius:4px;background:{color}12'>"
            f"{icon} {text}</div>", unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Action plan ───────────────────────────────────────────────────────────
    st.subheader("🚀 Action Plan")
    actions = [
        ("#F44336", "🔴 HIGH",   "Bangun sistem early-warning otomatis yang memonitor nilai & jumlah MK lulus di Sem 1 — trigger alert ke dosen wali jika di bawah threshold."),
        ("#F44336", "🔴 HIGH",   "Identifikasi mahasiswa dengan SPP tunggak di minggu ke-4 setiap semester, tawarkan cicilan atau beasiswa darurat sebelum semester berakhir."),
        ("#F44336", "🔴 HIGH",   "Program konseling khusus untuk mahasiswa berstatus debtor — hubungkan ke sumber beasiswa eksternal dan manajemen keuangan."),
        ("#FF9800", "🟡 MEDIUM", "Buat program pendampingan untuk mahasiswa non-tradisional (usia >23) dengan jadwal fleksibel dan mentoring karier."),
        ("#FF9800", "🟡 MEDIUM", "Perluas kuota beasiswa internal dan sosialisasikan beasiswa eksternal lebih agresif ke mahasiswa at-risk."),
        ("#4CAF50", "🟢 LOW",    f"Integrasikan model {best_name} ke Sistem Informasi Akademik (SIAK) sebagai risk-score real-time per mahasiswa per semester."),
    ]
    for color, priority, text in actions:
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:10px 16px;"
            f"margin-bottom:8px;border-radius:4px;background:{color}12'>"
            f"<strong style='color:{color}'>{priority}</strong> — {text}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Roadmap gantt ─────────────────────────────────────────────────────────
    st.subheader("🗓️ Implementation Roadmap")
    roadmap = pd.DataFrame({
        "Initiative":    ["Early Warning System", "SPP Intervention",
                          "Debt Counselling",     "Non-trad Support",
                          "Scholarship Expansion", "SIAK Integration"],
        "Start":         [1, 1, 2, 3, 3, 5],
        "Duration":      [4, 4, 3, 3, 3, 3],
        "Priority":      ["High","High","High","Medium","Medium","Low"],
    })
    color_p = {"High": "#F44336", "Medium": "#FF9800", "Low": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(11, 4))
    for _, row in roadmap.iterrows():
        ax.barh(row["Initiative"], row["Duration"],
                left=row["Start"] - 1,
                color=color_p[row["Priority"]], alpha=0.85,
                edgecolor="white", height=0.5)
        ax.text(row["Start"] - 1 + row["Duration"]/2, row["Initiative"],
                row["Priority"], ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_xlim(0, 9)
    ax.set_title("Implementation Roadmap", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    patches = [mpatches.Patch(color=c, label=p) for p, c in color_p.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.info(
        f"💬 **Kesimpulan:** Dropout mahasiswa bisa diprediksi (binary: Graduate vs Dropout) "
        f"dengan **{best_name}** yang mencapai akurasi **{acc:.2%}**. "
        f"Intervensi dengan ROI tertinggi ada di sisi finansial — "
        f"SPP dan utang. Mengintegrasikan model ke SIAK memungkinkan konselor bertindak "
        f"sebelum mahasiswa benar-benar keluar."
    )