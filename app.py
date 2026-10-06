import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostClassifier
import shap

st.set_page_config(page_title="Project Delay Intelligence", page_icon="📊", layout="wide")

TARGET = "delay_flag"
MODEL_FEATURES = ["contract_value","planned_duration_days","project_type","location","client_type","project_complexity"]
CATEGORICAL = ["project_type","location","client_type","project_complexity"]

@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    p = Path("project_dataset.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def prepare_X(d):
    x = d[MODEL_FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna("UNKNOWN").astype(str)
    return x

@st.cache_resource
def fit_model(d):
    x, y = prepare_X(d), d[TARGET].astype(int)
    m = CatBoostClassifier(iterations=300, depth=4, learning_rate=.04,
        l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=False)
    m.fit(x, y, cat_features=CATEGORICAL)
    return m

def money(x): return f"{x:,.0f}"

st.title("📊 Project Delay Intelligence Dashboard")
st.caption("Portfolio analysis • New-project risk prediction • Explainable AI")

uploaded = st.sidebar.file_uploader("Upload project CSV", type=["csv"])
df = load_data(uploaded)
if df.empty:
    st.info("Upload the CSV or place it beside this app as project_dataset.csv.")
    st.stop()

needed = set(MODEL_FEATURES + [TARGET])
missing = needed - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
for c in ["planned_start_date","planned_end_date","actual_start_date","actual_end_date"]:
    if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")

st.sidebar.header("Portfolio Filters")
def filt(label, col):
    vals = sorted(df[col].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, vals, vals)
pt = filt("Project Type","project_type")
cx = filt("Complexity","project_complexity")
cl = filt("Client Type","client_type")
f = df[df.project_type.astype(str).isin(pt) & df.project_complexity.astype(str).isin(cx) & df.client_type.astype(str).isin(cl)].copy()

a,b,c,d,e = st.columns(5)
a.metric("Projects", f"{len(f):,}")
b.metric("Delay Rate", f"{f[TARGET].mean()*100:.1f}%")
c.metric("Delayed", f"{f[TARGET].sum():,}")
d.metric("Avg Delay Days", f"{f.loc[f[TARGET].eq(1),'delay_days'].mean():.1f}" if "delay_days" in f else "N/A")
e.metric("Portfolio Value", money(f.contract_value.sum()))

tab1, tab2, tab3 = st.tabs(["📈 Portfolio Analysis","🚨 New Project Risk","🔎 Explainability"])

with tab1:
    l,r=st.columns(2)
    g=f.groupby("project_complexity")[TARGET].agg(["mean","count"]).reset_index()
    g["delay_rate"]=g["mean"]*100
    l.plotly_chart(px.bar(g,x="project_complexity",y="delay_rate",text="delay_rate",
        title="Delay Rate by Complexity",labels={"delay_rate":"Delay Rate (%)"}).update_traces(texttemplate="%{text:.1f}%",textposition="outside"),use_container_width=True)
    g=f.groupby("project_type")[TARGET].agg(["mean","count"]).reset_index()
    g["delay_rate"]=g["mean"]*100
    r.plotly_chart(px.bar(g.sort_values("delay_rate"),x="delay_rate",y="project_type",orientation="h",
        text="delay_rate",title="Delay Rate by Project Type",labels={"delay_rate":"Delay Rate (%)"}).update_traces(texttemplate="%{text:.1f}%"),use_container_width=True)

    l,r=st.columns(2)
    r.plotly_chart(px.scatter(f,x="planned_duration_days",y="contract_value",color=TARGET,
        size="contract_value",hover_data=["project_id","project_type","project_complexity"],
        log_y=True,title="Contract Value vs Planned Duration"),use_container_width=True)
    g=f.groupby("location")[TARGET].mean().mul(100).sort_values(ascending=False).head(12).reset_index(name="delay_rate")
    l.plotly_chart(px.bar(g,x="delay_rate",y="location",orientation="h",text="delay_rate",
        title="Highest Delay-Rate Locations",labels={"delay_rate":"Delay Rate (%)"}).update_traces(texttemplate="%{text:.1f}%"),use_container_width=True)

    cols=[x for x in ["project_id","project_name","project_type","location","client_type","contract_value",
                      "planned_duration_days","project_complexity","project_status","delay_days","delay_flag"] if x in f]
    st.subheader("Project-Level Portfolio")
    st.dataframe(f[cols].sort_values("delay_flag",ascending=False),use_container_width=True)

with tab2:
    st.subheader("New Project Delay Prediction")
    st.write("Only information available before execution is used by the model.")
    model=fit_model(df)
    l,m,r=st.columns(3)
    ptype=l.selectbox("Project Type",sorted(df.project_type.dropna().astype(str).unique()))
    loc=l.selectbox("Location",sorted(df.location.dropna().astype(str).unique()))
    client=l.selectbox("Client Type",sorted(df.client_type.dropna().astype(str).unique()))
    comp=m.selectbox("Project Complexity",sorted(df.project_complexity.dropna().astype(str).unique()))
    duration=m.number_input("Planned Duration (days)",1,int(df.planned_duration_days.median()))
    value=r.number_input("Contract Value",0.0,float(df.contract_value.median()),step=100000.0)

    new=pd.DataFrame([{"contract_value":value,"planned_duration_days":duration,"project_type":ptype,
                       "location":loc,"client_type":client,"project_complexity":comp}])
    prob=float(model.predict_proba(new[MODEL_FEATURES])[:,1][0])
    risk="HIGH" if prob>=.60 else ("MEDIUM" if prob>=.35 else "LOW")
    x,y,z=st.columns(3)
    x.metric("Predicted Delay Probability",f"{prob*100:.1f}%")
    y.metric("Risk Level",risk)
    z.metric("Historical Delay Rate",f"{df[TARGET].mean()*100:.1f}%")
    st.progress(prob)

    explainer=shap.TreeExplainer(model)
    sv=np.asarray(explainer.shap_values(new[MODEL_FEATURES]))[0]
    ex=pd.DataFrame({"Feature":MODEL_FEATURES,"SHAP Impact":sv})
    ex["Direction"]=np.where(ex["SHAP Impact"]>0,"Increases delay risk","Reduces delay risk")
    ex["Absolute Impact"]=ex["SHAP Impact"].abs()
    ex=ex.sort_values("Absolute Impact",ascending=False)
    st.plotly_chart(px.bar(ex.sort_values("SHAP Impact"),x="SHAP Impact",y="Feature",
        color="Direction",orientation="h",title="Why the Model Predicts This Risk"),use_container_width=True)
    st.dataframe(ex[["Feature","SHAP Impact","Direction"]],use_container_width=True)
    st.info("SHAP explains model contribution, not causality.")

with tab3:
    st.subheader("Global Feature Importance")
    model=fit_model(df)
    imp=pd.DataFrame({"Feature":MODEL_FEATURES,"Importance":model.get_feature_importance()}).sort_values("Importance",ascending=False)
    st.plotly_chart(px.bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",
        title="Global Model Feature Importance"),use_container_width=True)
    st.markdown("""### Modeling governance
**Included:** contract value, planned duration, project type, location, client type, complexity.

**Excluded:** actual dates, actual duration, realized delay days and project status. These are post-outcome or execution-time fields and would cause target leakage for a new project.

**Caution:** only 150 projects are available. Use this as decision support/prototype evidence, not a production-calibrated probability. Add substantially more historical projects and use chronological validation before deployment.""")
