import streamlit as st
from NDD.NDD import predict_interaction, polypharmacy_analysis

# ---------------------------
# PAGE SETTINGS
# ---------------------------

st.set_page_config(
    page_title="Drug Interaction AI",
    page_icon="💊",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.background-symbols{
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
opacity:0.05;
font-size:50px;
z-index:-1;
display:flex;
flex-wrap:wrap;
justify-content:center;
align-items:center;
}

.title{
font-size:50px;
text-align:center;
font-weight:bold;
color:#00ffff;
text-shadow:0px 0px 20px cyan;
}

.card{
background:rgba(255,255,255,0.06);
padding:25px;
border-radius:15px;
box-shadow:0px 0px 25px rgba(0,255,255,0.2);
margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# BACKGROUND DRUG SYMBOLS
# ---------------------------

st.markdown("""
<div class="background-symbols">
💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊
💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊
💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊 💊
</div>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------

st.markdown('<div class="title">💊 AI Drug Interaction Prediction</div>', unsafe_allow_html=True)

st.write("")

# ==================================================
# DRUG PAIR INTERACTION
# ==================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🔬 Drug Pair Interaction")

col1, col2 = st.columns(2)

with col1:
    drug1 = st.text_input("Drug 1")

with col2:
    drug2 = st.text_input("Drug 2")

if st.button("Predict Interaction"):

    if drug1 == "" or drug2 == "":
        st.warning("Please enter both drugs")

    else:

        result, prob, acc, prec, sens, spec, mcc = predict_interaction(drug1, drug2)

        if "Detected" in result:
            st.error(f"{result} | Confidence: {prob:.2f}")
        else:
            st.success(f"{result} | Confidence: {prob:.2f}")

        st.subheader("Model Performance")

        c1,c2,c3,c4,c5 = st.columns(5)

        c1.metric("Accuracy", f"{acc:.2f}")
        c2.metric("Precision", f"{prec:.2f}")
        c3.metric("Sensitivity", f"{sens:.2f}")
        c4.metric("Specificity", f"{spec:.2f}")
        c5.metric("MCC", f"{mcc:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# POLYPHARMACY
# ==================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🧬 Polypharmacy Drug Interaction Analysis")

drug_input = st.text_input(
"Enter drugs separated by comma (Example: Aspirin, Warfarin, Ibuprofen)"
)

if st.button("Analyze Polypharmacy"):

    if drug_input == "":
        st.warning("Please enter drugs")

    else:

        drugs = [d.strip() for d in drug_input.split(",")]

        results, risk, most_risky = polypharmacy_analysis(drugs)

        st.subheader("Pairwise Interaction Scores")

        for r in results:
            st.write(f"💊 {r[0]} + {r[1]} → Risk Score: {r[2]:.2f}")

        st.subheader("Polypharmacy Risk Score")

        st.progress(min(risk,1.0))

        st.write(f"Overall Risk Score: **{risk:.2f}**")

        st.subheader("Most Dangerous Drug")

        st.error(most_risky)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")

st.markdown(
"""
<center>
AI Clinical Decision Support System<br>
Drug Interaction Prediction using Neural Networks
</center>
""",
unsafe_allow_html=True
)