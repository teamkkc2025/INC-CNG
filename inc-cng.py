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
 

def flexible_priority_sampling(df, priority_col, sort_col, sample_size_per_category=3):
    """
    Ultra-flexible sampling function that works with ANY priority column values
    Handles S1, S2, S3, High, Medium, Low, or any other values
    """
    # Create a copy and clean the data
    df_copy = df.copy()
   
    # Handle different data types in the priority column
    df_copy[priority_col] = df_copy[priority_col].fillna('Unknown')  # Replace NaN with 'Unknown'
    df_copy[priority_col] = df_copy[priority_col].astype(str).str.strip()  # Convert to string and strip whitespace
   
    # Remove empty strings
    df_copy = df_copy[df_copy[priority_col] != '']
   
    if len(df_copy) == 0:
        return pd.DataFrame(), {}
   
    # Get unique values and sort them for consistent ordering
    unique_priorities = sorted(df_copy[priority_col].unique())
   
    st.info(f"🎯 Found priority levels: {', '.join(unique_priorities)}")  # Show user what was found
   
    sample_results = []
    sampling_summary = {}
   
    for priority in unique_priorities:
        priority_subset = df_copy[df_copy[priority_col] == priority].copy()
       
        if len(priority_subset) == 0:
            continue
       
        # Try to sort by the specified column
        try:
            # Check if the sort column contains numeric data
            if priority_subset[sort_col].dtype in ['int64', 'float64']:
                sorted_subset = priority_subset.sort_values(by=sort_col, ascending=True, na_last=True)
            else:
                # For non-numeric columns, try to convert or sort as string
                try:
                    # Try to convert to numeric
                    priority_subset[sort_col + '_numeric'] = pd.to_numeric(priority_subset[sort_col], errors='coerce')
                    sorted_subset = priority_subset.sort_values(by=sort_col + '_numeric', ascending=True, na_last=True)
                    sorted_subset = sorted_subset.drop(columns=[sort_col + '_numeric'])
                except:
                    # Fall back to string sorting
                    sorted_subset = priority_subset.sort_values(by=sort_col, ascending=True, na_last=True)
        except Exception as e:
            st.warning(f"Could not sort by {sort_col} for priority {priority}. Using original order.")
            sorted_subset = priority_subset.copy()
       
        total_count = len(sorted_subset)
       
        # Sampling logic
        if total_count <= sample_size_per_category:
            # Take all records if we have fewer than requested
            sample = sorted_subset.copy()
            sample['Sample_Type'] = f'{priority}_All_{total_count}_records'
            sample_results.append(sample)
           
            sampling_summary[priority] = {
                'total_records': total_count,
                'samples_taken': total_count,
                'sampling_method': 'All records (insufficient for top/bottom split)'
            }
           
        elif total_count <= sample_size_per_category * 2:
            # Take all records but mark them appropriately
            sample = sorted_subset.copy()
            sample['Sample_Type'] = f'{priority}_All_{total_count}_records'
            sample_results.append(sample)
           
            sampling_summary[priority] = {
                'total_records': total_count,
                'samples_taken': total_count,
                'sampling_method': 'All records (close to threshold)'
            }
        else:
            # Take top N and bottom N
            bottom_sample = sorted_subset.head(sample_size_per_category).copy()
            top_sample = sorted_subset.tail(sample_size_per_category).copy()
           
            bottom_sample['Sample_Type'] = f'{priority}_Bottom_{sample_size_per_category}'
            top_sample['Sample_Type'] = f'{priority}_Top_{sample_size_per_category}'
           
            sample_results.append(bottom_sample)
            sample_results.append(top_sample)
           
            sampling_summary[priority] = {
                'total_records': total_count,
                'samples_taken': sample_size_per_category * 2,
                'sampling_method': f'Top {sample_size_per_category} and Bottom {sample_size_per_category}'
            }
   
    # Combine all samples
    if sample_results:
        final_sample = pd.concat(sample_results, ignore_index=True)
        return final_sample, sampling_summary
    else:
        return pd.DataFrame(), {}
 
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
 
           
            output = BytesIO()
            df_checked.to_excel(output, index=False)
            output.seek(0)
            st.download_button("📥 Download Full Data with Checks", data=output,
                               file_name="checked_change_management.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
    if st.session_state.df_checked is not None:
        st.subheader("📄 Full Data with Calculated Fields")
        st.dataframe(st.session_state.df_checked)
 
        st.subheader("🎯 Enhanced Priority-Based Sampling")
       
        # Check if we have a priority column
        priority_columns = [col for col in st.session_state.df_checked.columns if
                          any(keyword in col.lower() for keyword in ['priority', 'risk', 'severity', 'impact', 'urgency'])]
       
        if priority_columns:
            st.info("💡 Detected potential priority columns. Select the appropriate one for sampling.")
       
        priority_column = st.selectbox("Select Priority/Risk Column", st.session_state.df_checked.columns.tolist())
        sampling_column = st.selectbox("Select Column for Sampling (Numerical)",
                                     [col for col in st.session_state.df_checked.columns])
       
        sample_size_per_cat = st.number_input("Samples per Priority Level (Top + Bottom)",
                                            min_value=1, max_value=10, value=3, step=1)
       
        col1, col2 = st.columns(2)
       
        with col1:
            if st.button("🎲 Generate Priority-Based Sample", key="priority_sample_cm"):
                sample_df, summary = flexible_priority_sampling(
                    st.session_state.df_checked,
                    priority_column,
                    sampling_column,
                    sample_size_per_cat
                )
               
                if not sample_df.empty:
                    st.success(f"✅ Generated {len(sample_df)} sample records")
                   
                    # Display sampling summary
                    st.subheader("📈 Sampling Summary")
                    for priority, stats in summary.items():
                        st.write(f"**{priority} Priority:**")
                        st.write(f"  - Total Records: {stats['total_records']}")
                        st.write(f"  - Samples Taken: {stats['samples_taken']}")
                        st.write(f"  - Method: {stats['sampling_method']}")
                   
                    # Display sample data
                    st.subheader("📊 Priority-Based Sample Records")
                    st.dataframe(sample_df)
                   
                    # Download button for sample
                    sample_output = BytesIO()
                    with pd.ExcelWriter(sample_output, engine='xlsxwriter') as writer:
                        sample_df.to_excel(writer, sheet_name='Priority_Sample', index=False)
                       
                        # Create summary sheet
                        summary_df = pd.DataFrame.from_dict(summary, orient='index')
                        summary_df.to_excel(writer, sheet_name='Sampling_Summary')
                   
                    sample_output.seek(0)
                    st.download_button(
                        "📥 Download Priority-Based Sample",
                        data=sample_output,
                        file_name="priority_based_sample_change_mgmt.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("⚠️ No valid data found for sampling")
       
        with col2:
            # Original random sampling (kept for backward compatibility)
            st.subheader("🎲 Traditional Random Sampling")
            sample_size = st.number_input("Number of Random Samples", min_value=1, max_value=len(st.session_state.df_checked), value=5, step=1)
            method = st.selectbox("Sampling Method", ["Top N (Longest)", "Bottom N (Quickest)", "Random"])
 
            if method == "Top N (Longest)":
                sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=False).head(sample_size)
            elif method == "Bottom N (Quickest)":
                sample_df = st.session_state.df_checked.sort_values(by=sampling_column, ascending=True).head(sample_size)
            else:
                sample_df = st.session_state.df_checked.sample(n=sample_size, random_state=1)
 
            st.write("📊 Traditional Sample Records")
            st.dataframe(sample_df)
 
            sample_output = BytesIO()
            sample_df.to_excel(sample_output, index=False)
            sample_output.seek(0)
            st.download_button("📥 Download Traditional Sample", data=sample_output,
                               file_name="traditional_sample_requests.xlsx",
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
 
            # 🔁 Enhanced Priority-Based Sampling for Incidents
            st.subheader("🎯 Enhanced Priority-Based Sampling")
           
            # Check for priority columns
            priority_columns = [col for col in df.columns if
                              any(keyword in col.lower() for keyword in ['priority', 'risk', 'severity', 'impact', 'urgency'])]
           
            if priority_columns:
                st.info("💡 Detected potential priority columns. Select the appropriate one for sampling.")
           
            priority_column = st.selectbox("Select Priority/Risk Column for Incidents", df.columns.tolist())
           
            # Select numerical column for sorting
            sort_column = st.selectbox("Select Column for Sorting", df.columns.tolist())
           
            sample_size_incidents = st.number_input("Samples per Priority Level (Top + Bottom) - Incidents",
                                                  min_value=1, max_value=10, value=3, step=1)
           
            col1, col2 = st.columns(2)
           
            with col1:
                if st.button("🎲 Generate Priority-Based Sample for Incidents", key="priority_sample_incidents"):
                    sample_df, summary = flexible_priority_sampling(
                        df,
                        priority_column,
                        sort_column,
                        sample_size_incidents
                    )
                   
                    if not sample_df.empty:
                        st.success(f"✅ Generated {len(sample_df)} incident sample records")
                       
                        # Display sampling summary
                        st.subheader("📈 Incident Sampling Summary")
                        for priority, stats in summary.items():
                            st.write(f"**{priority} Priority:**")
                            st.write(f"  - Total Records: {stats['total_records']}")
                            st.write(f"  - Samples Taken: {stats['samples_taken']}")
                            st.write(f"  - Method: {stats['sampling_method']}")
                       
                        # Display sample data
                        st.subheader("📊 Priority-Based Incident Sample")
                        st.dataframe(sample_df, height=300, use_container_width=True)
                       
                        # Download button for sample
                        sample_buffer = BytesIO()
                        with pd.ExcelWriter(sample_buffer, engine='xlsxwriter') as writer:
                            sample_df.to_excel(writer, sheet_name='Priority_Sample', index=False)
                           
                            # Create summary sheet
                            summary_df = pd.DataFrame.from_dict(summary, orient='index')
                            summary_df.to_excel(writer, sheet_name='Sampling_Summary')
                       
                        sample_buffer.seek(0)
                        st.download_button(
                            "📥 Download Priority-Based Incident Sample",
                            data=sample_buffer.getvalue(),
                            file_name="priority_based_incident_sample.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("⚠️ No valid priority data found for incident sampling")
           
            with col2:
                # Traditional Random Sampling (kept for backward compatibility)
                st.subheader("🎲 Traditional Random Sampling")
                sample_size = st.number_input("Number of Random Samples", min_value=1, max_value=len(df), value=5)
                if st.button("Generate Traditional Incident Sample"):
                    sample_df = df.sample(n=sample_size, random_state=42)
                    st.dataframe(sample_df,height=300, use_container_width=True)
 
                    sample_buffer = BytesIO()
                    sample_df.to_csv(sample_buffer, index=False)
                    st.download_button("📥 Download Traditional Sample", data=sample_buffer.getvalue(),
                                       file_name="traditional_incident_sample.csv", mime="text/csv")
               
            st.subheader("⚠️ Risk Category Threshold Check")
            risk_col = st.selectbox("Select Risk Level Column", df.columns)
 
            if risk_col:
                # Use flexible approach - no pattern matching, just use all unique values
                df["Parsed_Risk_Level"] = df[risk_col].fillna('Unknown').astype(str).str.strip()
               
                # Show unique risk levels found
                unique_risks = df["Parsed_Risk_Level"].unique()
                st.info(f"🎯 Found risk levels: {', '.join(unique_risks)}")
 
                st.markdown("Define SLA thresholds (in days) for each risk level:")
 
                # Dynamic threshold creation based on found risk levels
                thresholds_start_resolved = {}
                thresholds_resolved_close = {}
               
                st.subheader("📅 Start-to-Resolved Thresholds")
                col1, col2 = st.columns(2)
               
                with col1:
                    for i, risk in enumerate(unique_risks):
                        if risk != 'Unknown':
                            thresholds_start_resolved[risk] = st.number_input(
                                f"{risk} - Start to Resolved (days)",
                                min_value=0,
                                value=1 if 'critical' in risk.lower() or 's1' in risk.lower() else
                                      2 if 'high' in risk.lower() or 's2' in risk.lower() else
                                      4 if 'medium' in risk.lower() or 's3' in risk.lower() else 6,
                                step=1,
                                key=f"start_resolved_{risk}"
                            )
               
                st.subheader("📅 Resolved-to-Close Thresholds")
                with col2:
                    for i, risk in enumerate(unique_risks):
                        if risk != 'Unknown':
                            thresholds_resolved_close[risk] = st.number_input(
                                f"{risk} - Resolved to Close (days)",
                                min_value=0,
                                value=1 if 'critical' in risk.lower() or 's1' in risk.lower() else
                                      1 if 'high' in risk.lower() or 's2' in risk.lower() else
                                      2 if 'medium' in risk.lower() or 's3' in risk.lower() else 3,
                                step=1,
                                key=f"resolved_close_{risk}"
                            )
 
                # Apply flexible threshold checking
                def exceeds_flexible_threshold(row):
                    risk = row["Parsed_Risk_Level"]
                    if risk == 'Unknown' or risk not in thresholds_start_resolved:
                        return False
                   
                    exceeds_start_resolved = (
                        row["Start-Resolved"] is not None and
                        not pd.isna(row["Start-Resolved"]) and
                        row["Start-Resolved"] > thresholds_start_resolved[risk]
                    )
                   
                    exceeds_resolved_close = (
                        row["Resolved-Close"] is not None and
                        not pd.isna(row["Resolved-Close"]) and
                        row["Resolved-Close"] > thresholds_resolved_close[risk]
                    )
                   
                    return exceeds_start_resolved or exceeds_resolved_close
 
                df["Exceeds_Threshold"] = df.apply(exceeds_flexible_threshold, axis=1)
                observations_df = df[df["Exceeds_Threshold"] == True]
 
              
               
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
