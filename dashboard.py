import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MATPLOTLIB / SEABORN THEME
# ─────────────────────────────────────────────
BG_COLOR = "#0f1117"
AX_COLOR = "#1a1f2e"
TEXT_COL = "#c9d1e0"
GRID_COL = "#1e2a45"
PALETTE  = {"Stayed": "#4C9BE8", "Left": "#E85C5C"}
ACCENT   = ["#4C9BE8", "#E85C5C", "#56C68A", "#F0A84E", "#9B8FE8", "#4DD4CC"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_COLOR)
    ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        spine.set_linewidth(0.5)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="--")
    return ax

def make_fig(ncols=1, nrows=1, h=4, w=6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h))
    fig.patch.set_facecolor(BG_COLOR)
    return fig, axes

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/employee_data.csv")
    df["Attrition"] = pd.to_numeric(df["Attrition"], errors="coerce")
    return df

@st.cache_resource
def load_artifacts():
    try:
        model           = joblib.load("model/rf_model_attrition.pkl")
        scaler          = joblib.load("model/scaler.pkl")
        encoded_columns = joblib.load("model/encoded_columns.pkl")
        return model, scaler, encoded_columns
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None, None, None

df = load_data()
model, scaler, encoded_columns = load_artifacts()

df_clean = df.dropna(subset=["Attrition"]).copy()
df_clean["Attrition_Label"] = df_clean["Attrition"].map({1.0: "Left", 0.0: "Stayed"})

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Employee Attrition Analysis")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Prediction", "Insights"])
    st.divider()
    st.markdown("**Filters**")
    departments    = ["All"] + sorted(df["Department"].dropna().unique())
    sel_dept       = st.selectbox("Department", departments)
    gender_opts    = ["All"] + sorted(df["Gender"].dropna().unique())
    sel_gender     = st.selectbox("Gender", gender_opts)
    age_range      = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (25, 50))
    overtime_filter = st.selectbox("OverTime", ["All", "Yes", "No"])
    st.divider()
    st.caption(f"Dataset: {len(df):,} rows · {df.shape[1]} cols")

# Apply filters
fdf = df_clean.copy()
if sel_dept != "All":
    fdf = fdf[fdf["Department"] == sel_dept]
if sel_gender != "All":
    fdf = fdf[fdf["Gender"] == sel_gender]
fdf = fdf[(fdf["Age"] >= age_range[0]) & (fdf["Age"] <= age_range[1])]
if overtime_filter != "All":
    fdf = fdf[fdf["OverTime"] == overtime_filter]

# ═══════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Employee Attrition Dashboard")

    if len(fdf) == 0:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── KPI metrics ───────────────────────────────────────────────
    total      = len(fdf)
    attr_rate  = fdf["Attrition"].mean() * 100
    avg_income = fdf["MonthlyIncome"].mean()
    avg_tenure = fdf["YearsAtCompany"].mean()
    ot_pct     = (fdf["OverTime"] == "Yes").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees",    f"{total:,}")
    c2.metric("Attrition Rate",     f"{attr_rate:.1f}%",
              delta="⚠ High" if attr_rate > 15 else "✓ Healthy",
              delta_color="inverse" if attr_rate > 15 else "normal")
    c3.metric("Avg Monthly Income", f"${avg_income:,.0f}")
    c4.metric("Avg Tenure (yrs)",   f"{avg_tenure:.1f}")
    c5.metric("OverTime Workers",   f"{ot_pct:.1f}%")

    st.divider()

    # ── Row 1: Attrition count + Department ──────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = make_fig()
        counts = fdf["Attrition_Label"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[PALETTE.get(k, ACCENT[0]) for k in counts.index],
                      edgecolor=BG_COLOR, linewidth=0.8, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3, f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", color=TEXT_COL, fontsize=9)
        style_ax(ax, ylabel="Employees")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"💡 Attrition rate is **{attr_rate:.1f}%** in the current segment.")

    with col2:
        st.subheader("Attrition Rate by Department")
        fig, ax = make_fig()
        dept_data = fdf.groupby("Department")["Attrition"].mean().sort_values(ascending=False) * 100
        bars = ax.bar(dept_data.index, dept_data.values,
                      color=ACCENT[:len(dept_data)],
                      edgecolor=BG_COLOR, linewidth=0.8, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", color=TEXT_COL, fontsize=9)
        ax.set_xticks(range(len(dept_data)))
        ax.set_xticklabels(dept_data.index, rotation=15, ha="right")
        style_ax(ax, ylabel="Attrition Rate (%)")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"💡 **{dept_data.idxmax()}** has the highest attrition at {dept_data.max():.1f}%.")

    st.divider()

    # ── Row 2: OverTime + Job Role ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OverTime vs Attrition")
        fig, ax = make_fig()
        sns.countplot(data=fdf, x="OverTime", hue="Attrition_Label",
                      palette=PALETTE, ax=ax, edgecolor=BG_COLOR, linewidth=0.8)
        style_ax(ax, xlabel="OverTime", ylabel="Employees")
        ax.legend(title="", facecolor=AX_COLOR, labelcolor=TEXT_COL, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        ot_yes = fdf[fdf["OverTime"] == "Yes"]["Attrition"].mean() * 100
        ot_no  = fdf[fdf["OverTime"] == "No"]["Attrition"].mean() * 100
        st.caption(f"💡 Overtime attrition: **{ot_yes:.1f}%** vs non-overtime: **{ot_no:.1f}%**.")

    with col2:
        st.subheader("Attrition Rate by Job Role")
        fig, ax = make_fig(h=4.5)
        role_data = fdf.groupby("JobRole")["Attrition"].mean().sort_values() * 100
        colors = [ACCENT[0] if v < 20 else ACCENT[1] for v in role_data.values]
        ax.barh(role_data.index, role_data.values, color=colors,
                edgecolor=BG_COLOR, linewidth=0.8, height=0.6)
        for i, v in enumerate(role_data.values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", color=TEXT_COL, fontsize=8)
        style_ax(ax, xlabel="Attrition Rate (%)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"💡 **{role_data.idxmax()}** has the highest role-level attrition at {role_data.max():.1f}%.")


    st.divider()

    # ── Feature importance ────────────────────────────────────────
    if model is not None and encoded_columns is not None and hasattr(model, "feature_importances_"):
        st.subheader("Top 15 Feature Importances")
        cols = [col for col in encoded_columns if col != "Attrition"]

        fi   = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True).tail(15)
        fig, ax = make_fig(h=4, w=10)
        colors = [ACCENT[0] if v < fi.max() * 0.6 else ACCENT[1] for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors, edgecolor=BG_COLOR, linewidth=0.8, height=0.6)
        for i, v in enumerate(fi.values):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", color=TEXT_COL, fontsize=8)
        style_ax(ax, xlabel="Importance Score")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"💡 **{fi.idxmax()}** is the most important feature driving attrition predictions.")

# ═══════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ═══════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.title("Predict Employee Attrition")

    if model is None or encoded_columns is None:
        st.error("⚠️ Model files not found. Ensure rf_model_attrition.pkl, scaler.pkl, and encoded_columns.pkl are present.")
        st.stop()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Personal & Tenure**")
        age           = st.slider("Age", 18, 65, int(df["Age"].median()))
        marital       = st.selectbox("Marital Status", sorted(df["MaritalStatus"].dropna().unique()))
        education     = st.slider("Education (1–5)", 1, 5, int(df["Education"].median()))
        edu_field     = st.selectbox("Education Field", sorted(df["EducationField"].dropna().unique()))
        gender        = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        total_working = st.slider("Total Working Years", 0, 40, int(df["TotalWorkingYears"].median()))
        years_company = st.slider("Years at Company", 0, 40, int(df["YearsAtCompany"].median()))
        years_role    = st.slider("Years in Current Role", 0, 20, int(df["YearsInCurrentRole"].median()))
        years_promo   = st.slider("Years Since Last Promotion", 0, 15, int(df["YearsSinceLastPromotion"].median()))
        years_manager = st.slider("Years with Current Manager", 0, 20, int(df["YearsWithCurrManager"].median()))

    with col_b:
        st.markdown("**Work Conditions**")
        department      = st.selectbox("Department", sorted(df["Department"].dropna().unique()))
        job_role        = st.selectbox("Job Role", sorted(df["JobRole"].dropna().unique()))
        job_level       = st.slider("Job Level (1–5)", 1, 5, int(df["JobLevel"].median()))
        job_involvement = st.slider("Job Involvement (1–4)", 1, 4, int(df["JobInvolvement"].median()))
        job_sat         = st.slider("Job Satisfaction (1–4)", 1, 4, int(df["JobSatisfaction"].median()))
        env_sat         = st.slider("Environment Satisfaction (1–4)", 1, 4, int(df["EnvironmentSatisfaction"].median()))
        rel_sat         = st.slider("Relationship Satisfaction (1–4)", 1, 4, int(df["RelationshipSatisfaction"].median()))
        wlb             = st.slider("Work Life Balance (1–4)", 1, 4, int(df["WorkLifeBalance"].median()))
        overtime        = st.selectbox("OverTime", ["Yes", "No"])
        business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].dropna().unique()))
        distance        = st.slider("Distance From Home (km)", 1, 30, int(df["DistanceFromHome"].median()))
        num_companies   = st.slider("Num Companies Worked", 0, 10, int(df["NumCompaniesWorked"].median()))

    with col_c:
        st.markdown("**Compensation**")
        income       = st.number_input("Monthly Income ($)", 1000, 25000, int(df["MonthlyIncome"].median()), step=100)
        daily_rate   = st.slider("Daily Rate", int(df["DailyRate"].min()), int(df["DailyRate"].max()), int(df["DailyRate"].median()))
        hourly_rate  = st.slider("Hourly Rate", int(df["HourlyRate"].min()), int(df["HourlyRate"].max()), int(df["HourlyRate"].median()))
        monthly_rate = st.slider("Monthly Rate", int(df["MonthlyRate"].min()), int(df["MonthlyRate"].max()), int(df["MonthlyRate"].median()))
        pct_hike     = st.slider("Percent Salary Hike", 10, 25, int(df["PercentSalaryHike"].median()))
        stock_option = st.slider("Stock Option Level (0–3)", 0, 3, int(df["StockOptionLevel"].median()))
        perf_rating  = st.slider("Performance Rating (1–4)", 1, 4, int(df["PerformanceRating"].median()))
        training     = st.slider("Training Times Last Year", 0, 6, int(df["TrainingTimesLastYear"].median()))

    def build_input():
        row = {
            "Age": age, "BusinessTravel": business_travel,
            "DailyRate": daily_rate, "Department": department,
            "DistanceFromHome": distance, "Education": education,
            "EducationField": edu_field, "EmployeeCount": 1,
            "EnvironmentSatisfaction": env_sat, "Gender": gender,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobRole": job_role,
            "JobSatisfaction": job_sat, "MaritalStatus": marital,
            "MonthlyIncome": income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "Over18": "Y",
            "OverTime": overtime, "PercentSalaryHike": pct_hike,
            "PerformanceRating": perf_rating, "RelationshipSatisfaction": rel_sat,
            "StandardHours": 80, "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working, "TrainingTimesLastYear": training,
            "WorkLifeBalance": wlb, "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager,
        }
        input_df = pd.DataFrame([row])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=encoded_columns, fill_value=0)
        input_df = input_df.drop(columns=["Attrition"], errors="ignore")
        return input_df

    st.divider()
    if st.button("🔍 Predict Attrition Risk", use_container_width=True):
        input_df = build_input()

        st.markdown("**Input Summary**")
        summary = {
            "Age": age, "Monthly Income": f"${income:,}", "Department": department,
            "Job Role": job_role, "OverTime": overtime, "Marital Status": marital,
            "Stock Option": stock_option, "Job Satisfaction": job_sat,
            "Env Satisfaction": env_sat, "Years at Company": years_company,
        }
        st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

        try:
            input_scaled = scaler.transform(input_df)
            pred     = model.predict(input_scaled)[0]
            prob     = model.predict_proba(input_scaled)[0][1]
            prob_pct = prob * 100

            if pred == 1:
                st.error(f"⚠️ High Risk of Attrition — **{prob_pct:.1f}%** probability of leaving")
            else:
                st.success(f"✅ Low Risk of Attrition — **{prob_pct:.1f}%** probability of leaving")

            st.progress(int(prob_pct), text=f"Attrition Probability: {prob_pct:.1f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Ensure scaler and encoded_columns are compatible with the training pipeline.")

elif page == "Insights":
    st.title("Insights & Recommendations")

    if len(fdf) == 0:
        st.warning("No data matches the current filters.")
        st.stop()

    st.subheader("Key Insights")

    attr_rate = fdf["Attrition"].mean() * 100
    dept_attr = fdf.groupby("Department")["Attrition"].mean() * 100
    top_dept  = dept_attr.idxmax()

    ot_yes = fdf[fdf["OverTime"] == "Yes"]["Attrition"].mean() * 100
    ot_no  = fdf[fdf["OverTime"] == "No"]["Attrition"].mean() * 100

    high_role = fdf.groupby("JobRole")["Attrition"].mean().idxmax()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        - Overall attrition rate is **{attr_rate:.1f}%**
        - Highest attrition department: **{top_dept}**
        - Overtime workers attrition: **{ot_yes:.1f}%**
        - Non-overtime attrition: **{ot_no:.1f}%**
        """)

    with col2:
        st.markdown(f"""
        - Highest risk job role: **{high_role}**
        - Lower income employees tend to leave more
        - Early-career employees show higher attrition
        - Frequent business travel increases attrition risk
        """)

    st.divider()

    # ─────────────────────────────
    # RECOMMENDATIONS
    # ─────────────────────────────
    st.subheader("Recommendations")

    st.markdown("""
    ### 1. Reduce Overtime Burnout
    - Monitor employees with excessive overtime
    - Introduce workload balancing or automation
    - Encourage better work-life balance policies

    ### 2. Improve Compensation Strategy
    - Review salary for high-risk employees
    - Provide performance-based incentives
    - Benchmark with industry standards

    ### 3. Focus on Early-Career Employees
    - Create mentorship programs
    - Provide clear career progression paths
    - Increase engagement in first 2–3 years

    ### 4. Department-Level Intervention
    - Investigate root cause in high attrition departments
    - Improve leadership and management practices
    - Conduct employee satisfaction surveys

    ### 5. Travel & Work Flexibility
    - Reduce excessive business travel
    - Offer hybrid/remote options where possible

    ### 6. Predictive HR Strategy
    - Use the prediction model to flag high-risk employees
    - Intervene early with retention programs
    """)