import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="California House Price Estimator",
                   page_icon="🏠", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(120deg, #0B3D91 0%, #006AFF 100%);
       padding: 1.8rem 2.2rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {color: #DCE8FF; margin: .4rem 0 0 0;}
.card {background: white; border: 1px solid #E3E8F0; border-radius: 16px;
       padding: 1.3rem 1.5rem; box-shadow: 0 2px 10px rgba(16,24,40,.06);}
.label {color: #667085; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;}
.big {font-size: 2.9rem; font-weight: 700; color: #0B3D91; line-height: 1.1; margin: .2rem 0;}
.mid {font-size: 1.7rem; font-weight: 650; color: #1A1A2E; margin: .2rem 0;}
.sub {color: #667085; font-size: .9rem;}
.pill {display: inline-block; padding: .2rem .7rem; border-radius: 999px;
       font-size: .8rem; font-weight: 600; margin-top: .5rem;}
.ok {background: #E6F6EC; color: #137333;}
.warn {background: #FEF3E2; color: #B35C00;}
</style>
""", unsafe_allow_html=True)

# ---------- load ----------
tree = joblib.load('house_model.joblib')
lin = joblib.load('linear_model.joblib')
scaler = joblib.load('scaler.joblib')
r = pd.read_csv('feature_ranges.csv', index_col=0)
cols = list(r.columns)

labels = {
    'MedInc': 'Median income (x $10,000)', 'HouseAge': 'Median house age (years)',
    'AveRooms': 'Average rooms per household', 'AveBedrms': 'Average bedrooms per household',
    'Population': 'Block population', 'AveOccup': 'Average occupants per household',
    'Latitude': 'Latitude', 'Longitude': 'Longitude',
}

PRESETS = {
    "Los Angeles":    {'Latitude': 34.05, 'Longitude': -118.25, 'MedInc': 3.5},
    "San Francisco":  {'Latitude': 37.80, 'Longitude': -122.40, 'MedInc': 5.0},
    "Central Valley": {'Latitude': 36.70, 'Longitude': -119.80, 'MedInc': 3.0},
    "San Diego":      {'Latitude': 32.72, 'Longitude': -117.16, 'MedInc': 4.0},
}

def apply_preset():
    for k, v in PRESETS[st.session_state['preset']].items():
        st.session_state[k] = float(v)

if 'initialised' not in st.session_state:
    for c in cols:
        st.session_state[c] = float(r.loc['mean', c])
    for k, v in PRESETS["Los Angeles"].items():
        st.session_state[k] = float(v)
    st.session_state['initialised'] = True

# ---------- sidebar: the "search filters" ----------
with st.sidebar:
    st.header("Describe the area")
    st.selectbox("Quick location", list(PRESETS), key='preset', on_change=apply_preset)
    with st.expander("Location", expanded=True):
        for c in ['Latitude', 'Longitude']:
            st.slider(labels[c], float(r.loc['min', c]), float(r.loc['max', c]), key=c)
    with st.expander("Neighbourhood", expanded=True):
        for c in ['MedInc', 'Population', 'AveOccup']:
            st.slider(labels[c], float(r.loc['min', c]), float(r.loc['max', c]), key=c)
    with st.expander("Housing", expanded=False):
        for c in ['HouseAge', 'AveRooms', 'AveBedrms']:
            st.slider(labels[c], float(r.loc['min', c]), float(r.loc['max', c]), key=c)

# ---------- live prediction (no button) ----------
row = pd.DataFrame([{c: st.session_state[c] for c in cols}])[cols]
p_tree = tree.predict(row)[0] * 100000
p_lin_raw = lin.predict(scaler.transform(row))[0] * 100000
p_lin = min(max(p_lin_raw, 0), 500000)
gap = abs(p_tree - p_lin)

# ---------- header ----------
st.markdown("""
<div class="hero">
  <h1>🏠 California House Price Estimator</h1>
  <p>Two machine-learning models, one set of inputs. Adjust the area on the left and watch both estimates update.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Estimate", "Model comparison", "About"])

# ---------- tab 1 ----------
with tab1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown(f"""
        <div class="card">
          <div class="label">Estimated median value (Decision Tree)</div>
          <div class="big">${p_tree:,.0f}</div>
          <div class="sub">Typical error of this model on test data: about $40,000</div>
        </div>""", unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        a.markdown(f"""<div class="card"><div class="label">Decision Tree</div>
            <div class="mid">${p_tree:,.0f}</div></div>""", unsafe_allow_html=True)
        b.markdown(f"""<div class="card"><div class="label">Linear Regression</div>
            <div class="mid">${p_lin:,.0f}</div></div>""", unsafe_allow_html=True)
        st.write("")
        if gap > 100000:
            st.markdown(f'<span class="pill warn">Models differ by ${gap:,.0f}. Treat with caution.</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill ok">Models differ by ${gap:,.0f}</span>',
                        unsafe_allow_html=True)
        if p_lin != p_lin_raw:
            st.caption(f"Linear Regression's raw output (${p_lin_raw:,.0f}) was outside the "
                       "possible range, so it was clipped.")
    with right:
        st.map(pd.DataFrame({'lat': [st.session_state['Latitude']],
                             'lon': [st.session_state['Longitude']]}), zoom=8)
    st.caption("1990 census data. Values are for a neighbourhood block, not one house, and are capped at $500,000.")

# ---------- tab 2 ----------
with tab2:
    st.subheader("Which model makes smaller errors?")
    st.write("Mean absolute error on the test set, by longitude band, west to east. Lower is better.")
    bands = pd.DataFrame({
        "Longitude band": ["-124.2 to -122.6", "-122.6 to -121.0", "-121.0 to -119.4",
                           "-119.4 to -117.8", "-117.8 to -116.2", "-116.2 to -114.6"],
        "Blocks": [127, 1269, 355, 1643, 695, 39],
        "Decision Tree": [34681, 41364, 41982, 41103, 39119, 29294],
        "Linear Regression": [41226, 53588, 57756, 56077, 47374, 33420],
    })
    st.bar_chart(bands.set_index("Longitude band")[["Decision Tree", "Linear Regression"]])
    st.dataframe(bands, hide_index=True, use_container_width=True)
    st.caption("The tree had the lower error in every band. Both are still off by tens of thousands "
               "of dollars. One train-test split.")
    st.subheader("What the Decision Tree relies on")
    st.bar_chart(pd.Series(tree.feature_importances_, index=cols).sort_values())

# ---------- tab 3 ----------
with tab3:
    st.subheader("How this works")
    st.write("1. Models are trained in a Colab notebook on the California Housing dataset.")
    st.write("2. The trained models are saved as files with joblib.")
    st.write("3. This app loads those files and runs them on the values you choose.")
    st.subheader("Limits")
    st.write("- The data is from 1990, so prices are not current.")
    st.write("- Each prediction is for a block group, not a single house.")
    st.write("- Latitude and longitude are independent sliders, so impossible locations (like the ocean) can be chosen.")
    st.write("- This is a demo of a model comparison, not a production system.")