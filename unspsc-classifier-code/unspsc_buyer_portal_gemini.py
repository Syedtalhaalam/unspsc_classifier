import streamlit as st
import pandas as pd
import io
import re
from typing import List, Dict, Any

# --- Configuration ---
# NOTE: Update the file extension here
CLASSIFICATION_FILE = "results.xlsx" 
CONFIRMED_FILE = "confirmed_unspsc_output.csv"
UNMATCHED_CODE = "99999999"
UNMATCHED_COMMODITY = "No Match Found/Requires Expert Review"

# --- Helper Functions for Data Handling ---

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load and prepare the classification results data (now supports Excel)."""
    try:
        # --- CHANGE HERE: Use read_excel for .xlsx files ---
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            # Assuming the data is on the first sheet (Sheet1)
            df = pd.read_excel(file_path, sheet_name='Sheet1', keep_default_na=False)
        else:
            # Fallback for CSV
            df = pd.read_csv(file_path, keep_default_na=False)
            
        # Ensure key columns are strings and fill NA for consistent search
        df['Material_Code'] = df['Material_Code'].astype(str)
        df['Material_Description'] = df['Material_Description'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure '{file_path}' is correct and the sheet name is 'Sheet1'.")
        return pd.DataFrame()

def parse_recommendation_string(rec_string: str) -> List[Dict[str, str]]:
    """Parses recommendation strings like 'CODE - Commodity (conf: X.XX, LLM)' into a list of dictionaries."""
    if not rec_string:
        return []
    # Regex to find: CODE - COMMODITY (conf: X.XX, LLM)
    pattern = re.compile(r"(\d{8}) - (.+?)\s+\(conf:\s*(\d\.\d+)[^\)]*\)")
    
    # Check for simple format (e.g., from Final_Recommendation)
    simple_pattern = re.compile(r"(\d{8}) - (.+)")

    recommendations = []
    
    # Try parsing complex ranked string first
    if pattern.search(rec_string):
        matches = pattern.findall(rec_string)
        for code, commodity, confidence in matches:
            recommendations.append({
                'code': code,
                'commodity': commodity.strip(),
                'confidence': float(confidence)
            })
    # Fallback for simpler strings (like Top_5)
    elif simple_pattern.match(rec_string.split('\n')[0].strip()):
        for line in rec_string.split('\n'):
            line = line.strip()
            if simple_pattern.match(line):
                code, commodity = line.split(" - ", 1)
                recommendations.append({
                    'code': code.strip(),
                    'commodity': commodity.strip(),
                    'confidence': 0.0 # Placeholder
                })
    return recommendations

def mock_pipeline_run(material_code: str, material_description: str) -> Dict[str, Any]:
    """Mocks the pipeline's classification for a new material (Scenario 2)."""
    # Simple logic based on keywords for demonstration
    desc = material_description.lower()
    
    if 'breaker' in desc and '160a' in desc and 'lt' in desc:
        final_code = "39121616"
        final_comm = "Molded case circuit breakers"
        top5 = f"{final_code} - {final_comm}\n39121602 - Miniature circuit breakers\n39121601 - Circuit breakers"
        all30 = top5 + "\n26121613 - Insulated or covered cable\n30131500 - Blocks"
    elif 'cable' in desc and 'xlpe' in desc:
        final_code = "26121608"
        final_comm = "Aerial cable"
        top5 = f"{final_code} - {final_comm}\n26121629 - Power cable\n26121613 - Insulated or covered cable"
        all30 = top5 + "\n39121428 - Electrical connectors\n26121600 - Electrical cable and accessories"
    else:
        final_code = "30111700"
        final_comm = "Gasket and sealing material"
        top5 = f"{final_code} - {final_comm}"
        all30 = top5

    return {
        'Material_Code': material_code,
        'Material_Description': material_description,
        'Final_Recommendation': f"{final_code} - {final_comm}",
        'Top_5_Recommendations': top5,
        'All_Commodities_Ranked': all30,
        'Decision_Rule': 'MOCKED_LLM_RUN',
        'Confidence': 0.99
    }

# --- Streamlit App Initialization ---

st.set_page_config(
    page_title="UNSPSC Classification Portal",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize Session State
if 'confirmed_data' not in st.session_state:
    st.session_state['confirmed_data'] = pd.DataFrame(columns=[
        'Date_Confirmed', 'Confirmed_By', 'Material_Code', 'Material_Description',
        'Pipeline_Final_Recommendation', 'Confirmed_UNSPSC_Code', 'Confirmed_Commodity', 'Buyer_Comment'
    ])

if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""

if 'selected_material_data' not in st.session_state:
    st.session_state['selected_material_data'] = None

if 'show_level' not in st.session_state:
    st.session_state['show_level'] = 0 # 0: Final, 1: Top_5, 2: All_Commodities

# --- Main App Body ---

st.title("⚡ UNSPSC Classification & Confirmation Portal")

# Load data
df_raw = load_data(CLASSIFICATION_FILE)

if df_raw.empty:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("App Context")
    st.info(f"Loaded **{len(df_raw)}** existing material classifications from `{CLASSIFICATION_FILE}`.")
    st.write("This portal enables Procurement Buyers to confirm or flag UNSPSC codes recommended by the LLM classification pipeline.")
    st.text_input("Buyer ID / User Name", value="BUYER-001", key="buyer_id")
    st.caption("Confirmed data is saved to the **Reporting** tab.")

# --- Tab Setup ---
tab1, tab2, tab3 = st.tabs(["1. 📝 Buyer Confirmation (Existing Materials)", "2. ✨ New Material Search (Pipeline)", "3. 📊 Reporting & Download"])

# --- Tab 1: Buyer Confirmation (Existing Materials) ---
with tab1:
    st.header("Buyer Confirmation - Pipeline Results")
    
    # --- Search and Filter ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.search_query = st.text_input(
            "Search Material Code or Description",
            value=st.session_state.search_query,
            placeholder="e.g., 10000355 or BASE; FUSE HRC LT"
        )
    
    # Filter logic
    if st.session_state.search_query:
        query = st.session_state.search_query.upper()
        df_filtered = df_raw[
            df_raw['Material_Code'].str.contains(query, na=False) |
            df_raw['Material_Description'].str.contains(query, na=False)
        ]
    else:
        df_filtered = df_raw.head(0) # Show nothing if no search
    
    # Selection
    st.subheader(f"Found {len(df_filtered)} matching materials:")
    if not df_filtered.empty:
        # Use .head(50) for performance on large files
        df_display = df_filtered[['Material_Code', 'Material_Description', 'Final_Recommendation', 'Commodity_Code']].head(50) 
        
        selected_index = st.data_editor(
            df_display,
            column_config={
                "Material_Code": st.column_config.TextColumn("Material Code"),
                "Material_Description": st.column_config.TextColumn("Material Description"),
                "Final_Recommendation": st.column_config.TextColumn("Pipeline Recommendation (Code - Commodity)"),
                "Commodity_Code": st.column_config.TextColumn("Pipeline Best Code")
            },
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            on_change=lambda: st.session_state.update({'show_level': 0}),
            key="data_editor"
        )

        if selected_index:
            # Get the selected row's data
            idx = selected_index.index[0]
            selected_material_data = df_filtered.loc[idx].to_dict()
            st.session_state['selected_material_data'] = selected_material_data
            st.markdown("---")
            st.subheader(f"Confirmation for: **{selected_material_data['Material_Code']}**")

            # --- Confirmation UI ---
            current_code_commodity = st.session_state['selected_material_data']['Final_Recommendation']
            
            # --- Recommendation Section ---
            rec_options = []
            
            # 1. Final Recommendation
            final_code, final_comm = current_code_commodity.split(" - ", 1)
            rec_options.append((final_code.strip(), final_comm.strip()))

            # 2. Top_5 Recommendations
            if st.session_state.show_level >= 1 and st.session_state['selected_material_data']['Top_5_Recommendations']:
                st.info("Showing additional recommendations (Level 1: Top 5 List)")
                top5_recs = parse_recommendation_string(st.session_state['selected_material_data']['Top_5_Recommendations'])
                for rec in top5_recs:
                    if (rec['code'], rec['commodity']) not in [(c, d) for c, d in rec_options]:
                        rec_options.append((rec['code'], rec['commodity']))
            
            # 3. All_Commodities_Ranked
            if st.session_state.show_level == 2 and st.session_state['selected_material_data']['All_Commodities_Ranked']:
                st.warning("Showing all ranked recommendations (Level 2: All 30 Candidates)")
                all_recs = parse_recommendation_string(st.session_state['selected_material_data']['All_Commodities_Ranked'])
                for rec in all_recs:
                    if (rec['code'], rec['commodity']) not in [(c, d) for c, d in rec_options]:
                        rec_options.append((rec['code'], rec['commodity']))

            # Add fallback option
            rec_options.append((UNMATCHED_CODE, UNMATCHED_COMMODITY))

            # Create display list for dropdown
            display_options = [f"{code} - {comm}" for code, comm in rec_options]
            
            # Pre-select the Pipeline's Final Recommendation
            default_index = display_options.index(f"{final_code} - {final_comm}")
            
            # Dropdown for selection
            selected_option_str = st.selectbox(
                "Select/Confirm the Final UNSPSC Code:",
                options=display_options,
                index=default_index,
                key="unspsc_selector"
            )

            # --- Action Buttons ---
            col_a, col_b, col_c = st.columns([1, 1, 3])

            def confirm_material():
                """Logic to save the confirmed code to session state."""
                try:
                    confirmed_code, confirmed_comm = st.session_state.unspsc_selector.split(" - ", 1)
                except ValueError:
                    confirmed_code, confirmed_comm = UNMATCHED_CODE, UNMATCHED_COMMODITY

                new_row = {
                    'Date_Confirmed': pd.to_datetime('today').strftime("%Y-%m-%d"),
                    'Confirmed_By': st.session_state.buyer_id,
                    'Material_Code': st.session_state['selected_material_data']['Material_Code'],
                    'Material_Description': st.session_state['selected_material_data']['Material_Description'],
                    'Pipeline_Final_Recommendation': st.session_state['selected_material_data']['Final_Recommendation'],
                    'Confirmed_UNSPSC_Code': confirmed_code,
                    'Confirmed_Commodity': confirmed_comm,
                    'Buyer_Comment': st.session_state.buyer_comments
                }
                
                # Append to the confirmed_data DataFrame in session state
                st.session_state.confirmed_data = pd.concat([st.session_state.confirmed_data, pd.Series(new_row).to_frame().T], ignore_index=True)
                
                st.success(f"Material **{new_row['Material_Code']}** confirmed with UNSPSC **{confirmed_code}** and saved! Search for the next material.")
                # Reset UI state
                st.session_state.update({
                    'search_query': "",
                    'selected_material_data': None,
                    'show_level': 0,
                    'unspsc_selector': None,
                    'buyer_comments': ""
                })

            with col_a:
                st.button("✅ Confirm & Save", on_click=confirm_material, type="primary")

            with col_b:
                if st.session_state.show_level < 2:
                    st.button(
                        f"➡️ More Recommendations (Level {st.session_state.show_level + 1})",
                        on_click=lambda: st.session_state.update({'show_level': st.session_state.show_level + 1})
                    )
                else:
                    st.caption("All recommendations shown.")

            with col_c:
                st.text_area("Buyer Comments / Disagreement Reason", key="buyer_comments", height=50)

# --- Tab 2: New Material Search (Pipeline) ---
with tab2:
    st.header("New Material Classification Search")
    st.info("This section simulates running the LLM classification pipeline on a brand new material description.")
    
    new_matl_code = st.text_input("New Material Code (Optional)", key="new_matl_code")
    new_matl_desc = st.text_area("New Material Description", placeholder="e.g., BREAKER; LT 160A WITH METAL PLATE", key="new_matl_desc")
    
    if st.button("✨ Run Classification Pipeline", type="primary", disabled=not new_matl_desc):
        with st.spinner(f"Running LLM classification for '{new_matl_desc[:40]}...'"):
            # Mock the complex pipeline execution
            classification_results = mock_pipeline_run(new_matl_code, new_matl_desc)

            st.markdown("---")
            st.subheader("Classification Results")
            
            st.metric(
                label="Final Recommended UNSPSC Code & Commodity",
                value=classification_results['Final_Recommendation'],
                delta="Confidence: 99%" # Mocked Confidence
            )

            # Display all recommendations in expandable sections
            with st.expander("View Top 5 Recommendations"):
                st.text(classification_results['Top_5_Recommendations'])

            with st.expander("View All Ranked Recommendations"):
                st.text(classification_results['All_Commodities_Ranked'])
            
            st.success("Classification complete. The results are available above.")


# --- Tab 3: Reporting & Download ---
with tab3:
    st.header("Confirmed UNSPSC Codes Report")
    
    df_confirmed = st.session_state.confirmed_data
    
    if df_confirmed.empty:
        st.warning("No materials have been confirmed yet.")
    else:
        st.subheader(f"Total Confirmed Materials: {len(df_confirmed)}")
        
        # Display table
        st.dataframe(df_confirmed, use_container_width=True, hide_index=True)
        
        # Download Button (Enterprise Grade Feature)
        @st.cache_data
        def convert_df_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        @st.cache_data
        def convert_df_to_excel(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Confirmed UNSPSC')
            return output.getvalue()
        
        csv_data = convert_df_to_csv(df_confirmed)
        excel_data = convert_df_to_excel(df_confirmed)

        st.markdown("---")
        
        col_d, col_e = st.columns(2)
        
        with col_d:
            st.download_button(
                label=f"⬇️ Download Report as CSV ({len(df_confirmed)} rows)",
                data=csv_data,
                file_name=CONFIRMED_FILE,
                mime='text/csv',
                type="secondary"
            )
        
        with col_e:
            st.download_button(
                label=f"⬇️ Download Report as Excel ({len(df_confirmed)} rows)",
                data=excel_data,
                file_name=CONFIRMED_FILE.replace(".csv", ".xlsx"),
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                type="secondary"
            )