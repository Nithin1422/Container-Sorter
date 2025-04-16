
import streamlit as st
import pandas as pd
from io import BytesIO
from model import run_optimization

st.set_page_config(page_title="Container Optimizer", layout="wide")
st.title("📦 Container Optimization Tool")
st.markdown("Upload your SKU sheet and get the best container + optimal fill plan 🚛")

# File Upload
uploaded_file = st.file_uploader("Upload your SKU input file (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("🔍 Input Preview")
    st.dataframe(df)

    if st.button("🚀 Run Optimization"):
        container, final_df, summary_df = run_optimization(df)

        st.success(f"✅ Best Container Selected: **{container}**")
        st.subheader("📋 Final SKU Mix")
        st.dataframe(final_df)

        st.subheader("📊 Utilization Summary")
        st.bar_chart(summary_df.set_index("Metric"))

        # Excel export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, index=False, sheet_name="Optimized Mix")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
        output.seek(0)

        st.download_button(
            label="⬇️ Download Result Excel",
            data=output,
            file_name="optimized_container_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👈 Please upload an Excel file to get started.")
