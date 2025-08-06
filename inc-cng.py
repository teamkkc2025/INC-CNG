import streamlit as st
import pandas as pd
from io import BytesIO
import io
import datetime
import random
import openai
from typing import Optional
 
st.set_page_config(page_title="ITGC Application", layout="wide")
st.title("📊 ITGC Application")
 
# Initialize session state for API key
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
 
# API Key Configuration Section
st.sidebar.header("🤖 AI Configuration")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    value=st.session_state.groq_api_key,
    help="Get your API key from https://console.groq.com/keys"
)
 
if api_key:
    st.session_state.groq_api_key = api_key
    st.sidebar.success("✅ API Key configured!")
 
# Helper function for AI integration
def get_ai_summary(prompt: str, api_key: str) -> Optional[str]:
    """
    Generate AI summary using Groq API (OpenAI-compatible)
    """
    if not api_key:
        return "⚠️ Please configure your Groq API key in the sidebar to use AI features."
   
    try:
        # Configure OpenAI client for Groq
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
       
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert IT auditor specializing in ITGC (IT General Controls) analysis. Provide professional, concise insights about audit findings, potential root causes, and recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
       
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"
 
def create_ai_insight_section(findings_summary: str, module_name: str):
    """
    Create AI insight section with button trigger
    """
    if st.button(f"🧠 Generate AI Insights for {module_name}", key=f"ai_button_{module_name}"):
        if not st.session_state.groq_api_key:
            st.error("Please configure your Groq API key in the sidebar first.")
            return
           
        with st.spinner("🤖 Generating AI insights..."):
            prompt = f"""
            Analyze the following ITGC {module_name} findings and provide:
            1. A brief summary of key findings
            2. Possible root causes for any issues identified
            3. A sample audit observation or comment
            4. Recommendations for improvement
           
            Findings:
            {findings_summary}
           
            If no significant issues are found, acknowledge this and suggest proactive monitoring approaches.
            """
           
            ai_response = get_ai_summary(prompt, st.session_state.groq_api_key)
           
            with st.expander("🧠 AI-Generated Insights", expanded=True):
                st.markdown(ai_response)
 
# User selection
module = st.radio("Select Module", ["Incident Management", "Change Management"])
 
# -------------------------
# 🔁 CHANGE MANAGEMENT FLOW
# -------------------------
if module == "Change Management":
    uploaded_file = st.file_uploader("Upload Change Management File (CSV or Excel)", type=["csv", "xlsx"])
 
    if "df_checked" not in st.session_state:
        st.session_state.df_checked = None
 
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
 
        st.subheader("Select Relevant Columns")
        columns = df.columns.tolist()
        col_request_id = st.selectbox("Request ID Column", columns)
        columns_with_none = ["None"] + columns
        col_raised_date = st.selectbox("Raised Date Column", columns_with_none)
        col_resolved_date = st.selectbox("Resolved Date Column", columns_with_none)
 
        if st.button("Run Check"):
            df_checked = df.copy()
            df_checked.rename(columns={col_request_id: "request_id"}, inplace=True)
 
            if col_raised_date != "None":
                df_checked.rename(columns={col_raised_date: "raised_date"}, inplace=True)
                df_checked["raised_date"] = pd.to_datetime(df_checked["raised_date"], errors='coerce')
                df_checked["missing_raised"] = df_checked["raised_date"].isna()
            else:
                df_checked["raised_date"] = pd.NaT
                df_checked["missing_raised"] = False
 
            if col_resolved_date != "None":
                df_checked.rename(columns={col_resolved_date: "resolved_date"}, inplace=True)
                df_checked["resolved_date"] = pd.to_datetime(df_checked["resolved_date"], errors='coerce')
                df_checked["missing_resolved"] = df_checked["resolved_date"].isna()
            else:
                df_checked["resolved_date"] = pd.NaT
                df_checked["missing_resolved"] = False
 
            if col_raised_date != "None" and col_resolved_date != "None":
                df_checked["resolved_before_raised"] = df_checked["resolved_date"] < df_checked["raised_date"]
                df_checked["days_to_resolve"] = (df_checked["resolved_date"] - df_checked["raised_date"]).dt.days
            else:
                df_checked["resolved_before_raised"] = False
                df_checked["days_to_resolve"] = None
 
            st.session_state.df_checked = df_checked
 
            st.subheader("📊 Summary of Findings")
            missing_raised = df_checked['missing_raised'].sum()
            missing_resolved = df_checked['missing_resolved'].sum()
            resolved_before_raised = df_checked['resolved_before_raised'].sum()
           
            st.write(f"Missing Raised Dates: {missing_raised}")
            st.write(f"Missing Resolved Dates: {missing_resolved}")
            st.write(f"Resolved Before Raised: {resolved_before_raised}")
 
            # AI Insights Section for Change Management
            findings_summary = f"""
            Change Management Analysis Results:
            - Total Records: {len(df_checked)}
            - Missing Raised Dates: {missing_raised}
            - Missing Resolved Dates: {missing_resolved}
            - Resolved Before Raised (Data Quality Issue): {resolved_before_raised}
            - Average Days to Resolve: {df_checked['days_to_resolve'].mean():.1f if df_checked['days_to_resolve'].notna().any() else 'N/A'}
            """
           
            create_ai_insight_section(findings_summary, "Change Management")
 
            output = BytesIO()
            df_checked.to_excel(output, index=False)
            output.seek(0)
            st.download_button("📥 Download Full Data with Checks", data=output,
                               file_name="checked_change_management.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
    if st.session_state.df_checked is not None:
        st.subheader("📄 Full Data with Calculated Fields")
        st.dataframe(st.session_state.df_checked)
 
        st.subheader("🎯 Sampling Section")
        sampling_column = st.selectbox("Select Column for Sampling", st.session_state.df_checked.columns.tolist())
        sample_size = st.number_input("Number of Samples", min_value=1, max_value=len(st.session_state.df_checked), value=5, step=1)
        method = st.selectbox("Sampling Method", ["Top N (Longest)", "Bottom N (Quickest)", "Random"])
 
        if method == "Top N (Longest)":
            sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=False).head(sample_size)
        elif method == "Bottom N (Quickest)":
            sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=True).head(sample_size)
        else:
            sample_df = st.session_state.df_checked.sample(n=sample_size, random_state=1)
 
        st.write("📊 Sampled Records")
        st.dataframe(sample_df)
 
        sample_output = BytesIO()
        sample_df.to_excel(sample_output, index=False)
        sample_output.seek(0)
        st.download_button("📥 Download Sample Records", data=sample_output,
                           file_name="sampled_requests.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
# -------------------------
# 🧯 INCIDENT MANAGEMENT FLOW
# -------------------------
elif module == "Incident Management":
    uploaded_file = st.file_uploader("Upload Incident Management File (Excel or CSV)", type=["csv", "xlsx"])
 
    def load_data(uploaded_file):
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        return None
 
    def calculate_date_differences(df, start_col, end_col, resolved_col):
        if start_col != "None":
            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        if resolved_col != "None":
            df[resolved_col] = pd.to_datetime(df[resolved_col], errors='coerce')
        if end_col != "None":
            df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
 
        if start_col != "None" and resolved_col != "None":
            df['Start-Resolved'] = (df[resolved_col] - df[start_col]).dt.days
        else:
            df['Start-Resolved'] = None
 
        if resolved_col != "None" and end_col != "None":
            df['Resolved-Close'] = (df[end_col] - df[resolved_col]).dt.days
        else:
            df['Resolved-Close'] = None
 
        return df
 
    if uploaded_file:
        df = load_data(uploaded_file)
 
        if df is not None:
            st.subheader("📋 Incident Management Columns")
            st.write("Data preview:", df.head())
 
            columns_with_none = ["None"] + df.columns.tolist()
            start_col = st.selectbox("Select Start Date Column", columns_with_none)
            resolved_col = st.selectbox("Select Resolved Date Column", columns_with_none)
            end_col = st.selectbox("Select Close/End Date Column", columns_with_none)
 
            df = calculate_date_differences(df, start_col, end_col, resolved_col)
 
            st.write("✅ Updated Data with Date Differences:")
            st.dataframe(df,height=200, use_container_width=True)
 
            st.download_button("📥 Download Updated File", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="updated_incidents.csv", mime="text/csv")
 
            # 🔁 Random Sampling
            st.subheader("🎯 Random Sampling")
            sample_size = st.number_input("Number of Random Samples", min_value=1, max_value=len(df), value=5)
            if st.button("Generate Incident Sample"):
                sample_df = df.sample(n=sample_size, random_state=42)
                st.dataframe(sample_df,height=300, use_container_width=True)
 
                sample_buffer = BytesIO()
                sample_df.to_csv(sample_buffer, index=False)
                st.download_button("📥 Download Sample Records", data=sample_buffer.getvalue(),
                                   file_name="incident_sample.csv", mime="text/csv")
               
            st.subheader("⚠️ Risk Category Threshold Check")
            risk_col = st.selectbox("Select Risk Level Column", df.columns)
 
            if risk_col:
                # Extract last word (risk level) regardless of delimiter or format
                df["Parsed_Risk_Level"] = df[risk_col].astype(str).str.extract(r'([Cc]ritical|[Hh]igh|[Mm]edium|[Ll]ow)', expand=False).str.capitalize()
 
                st.markdown("Define SLA thresholds (in days) for each risk level:")
 
                # Start-Resolved thresholds
                crit_threshold = st.number_input("Critical Risk Threshold (Start-Resolved)", min_value=0, value=1)
                high_threshold = st.number_input("High Risk Threshold (Start-Resolved)", min_value=0, value=2)
               
                med_threshold = st.number_input("Medium Risk Threshold (Start-Resolved)", min_value=0, value=4)
                low_threshold = st.number_input("Low Risk Threshold (Start-Resolved)", min_value=0, value=6)
 
                # Resolved-Close thresholds
                crit_close_threshold = st.number_input("Critical Risk Threshold (Resolved-Close)", min_value=0, value=1)
                high_close_threshold = st.number_input("High Risk Threshold (Resolved-Close)", min_value=0, value=1)
                med_close_threshold = st.number_input("Medium Risk Threshold (Resolved-Close)", min_value=0, value=2)
                low_close_threshold = st.number_input("Low Risk Threshold (Resolved-Close)", min_value=0, value=3)
 
                # Apply filters
                def exceeds_threshold(row):
                    risk = row["Parsed_Risk_Level"]
                    if risk == "Critical":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > crit_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > crit_close_threshold)
                        )
                    elif risk == "High":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > high_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > high_close_threshold)
                        )
                    elif risk == "Medium":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > med_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > med_close_threshold)
                        )
                    elif risk == "Low":
                        return (
                            (row["Start-Resolved"] is not None and row["Start-Resolved"] > low_threshold) or
                            (row["Resolved-Close"] is not None and row["Resolved-Close"] > low_close_threshold)
                        )
                    return False
 
                df["Exceeds_Threshold"] = df.apply(exceeds_threshold, axis=1)
                observations_df = df[df["Exceeds_Threshold"] == True]
 
                # Incident Management AI Insights Section
                if not observations_df.empty:
                    st.warning(f"{len(observations_df)} record(s) exceeded the threshold limits.")
                    st.dataframe(observations_df, height=200, use_container_width=True)
 
                    obs_buffer = BytesIO()
                    observations_df.to_csv(obs_buffer, index=False)
                    st.download_button("📥 Download Observations File", data=obs_buffer.getvalue(),
                                    file_name="incident_observations.csv", mime="text/csv")
                   
                    # AI Insights for incidents with threshold violations
                    avg_start_resolved = df['Start-Resolved'].mean() if df['Start-Resolved'].notna().any() else 0
                    avg_resolved_close = df['Resolved-Close'].mean() if df['Resolved-Close'].notna().any() else 0
                   
                    findings_summary = f"""
                    Incident Management Analysis Results:
                    - Total Incidents: {len(df)}
                    - SLA Violations: {len(observations_df)}
                    - Violation Rate: {(len(observations_df)/len(df)*100):.1f}%
                    - Average Start-to-Resolved Time: {avg_start_resolved:.1f} days
                    - Average Resolved-to-Close Time: {avg_resolved_close:.1f} days
                    - Risk Level Distribution: {df['Parsed_Risk_Level'].value_counts().to_dict()}
                    - Most Common Violating Risk Level: {observations_df['Parsed_Risk_Level'].mode().iloc[0] if not observations_df.empty else 'N/A'}
                    """
                   
                else:
                    st.success("✅ All records are within threshold limits.")
                   
                    # AI Insights for compliant incidents
                    avg_start_resolved = df['Start-Resolved'].mean() if df['Start-Resolved'].notna().any() else 0
                    avg_resolved_close = df['Resolved-Close'].mean() if df['Resolved-Close'].notna().any() else 0
                   
                    findings_summary = f"""
                    Incident Management Analysis Results:
                    - Total Incidents: {len(df)}
                    - SLA Compliance: 100% (No violations found)
                    - Average Start-to-Resolved Time: {avg_start_resolved:.1f} days
                    - Average Resolved-to-Close Time: {avg_resolved_close:.1f} days
                    - Risk Level Distribution: {df['Parsed_Risk_Level'].value_counts().to_dict()}
                    - All incidents resolved within defined SLA thresholds
                    """
 
                create_ai_insight_section(findings_summary, "Incident Management")
               
                # ✅ Download full dataset with flags
                st.subheader("📥 Download Full Data with SLA Checks")
                full_buffer = BytesIO()
                with pd.ExcelWriter(full_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Full_Data")
                st.download_button("Download Full Incident Data", data=full_buffer.getvalue(),
                                file_name="incident_full_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
# -------------------------
# 👥 USER ACCESS MANAGEMENT FLOW (Placeholder for future implementation)
# -------------------------
elif module == "User Access Management":
    st.info("🚧 User Access Management module coming soon!")
    st.write("This module will include:")
    st.write("- User access reviews")
    st.write("- Segregation of duties analysis")
    st.write("- Privileged access monitoring")
    st.write("- AI-powered access risk assessment")
 
# Footer
st.markdown("---")
st.markdown("**ITGC Application with AI Insights** | Powered by Groq API")
