import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# 1. TRY TO IMPORT PIPELINE
# -------------------------------------------------------------------
PIPELINE_AVAILABLE = False
UNSPSCPipeline = None
ClassificationResult = None

try:
    # Make sure unspsc_classifier_v10.py is in the SAME folder as this file
    from unspsc_classifier_v10 import UNSPSCPipeline, ClassificationResult
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(
        "UNSPSC pipeline not available (unspsc_classifier_v10.py not imported). "
        "Scenario 2 (new material classification) will be disabled."
    )

# -------------------------------------------------------------------
# 2. PATHS & CONSTANTS
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "confirmed_results"
CACHE_DIR = BASE_DIR / "cache"

# results.xlsx MUST be placed next to this file (unspsc_buyer_portal.py)
RESULTS_FILE = BASE_DIR / "results.xlsx"

CONFIRMED_FILE = OUTPUT_DIR / "confirmed_materials.xlsx"
FLAGGED_FILE = OUTPUT_DIR / "flagged_materials.xlsx"

# Create folders if missing
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------


@st.cache_data
def load_results_data() -> Optional[pd.DataFrame]:
    """
    Load pre-classified materials from results.xlsx.
    This is the file produced by your offline UNSPSC pipeline.
    """
    if not RESULTS_FILE.exists():
        return None

    try:
        df = pd.read_excel(RESULTS_FILE)
        return df
    except Exception as e:
        st.error(f"Error reading {RESULTS_FILE.name}: {e}")
        return None


def parse_list_field(value) -> List[str]:
    """
    Normalize Top_5_Recommendations / All_Commodities_Ranked fields
    into a list of strings. Supports JSON, newline, pipe, or plain string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()
    if not text:
        return []

    # Try JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass

    # Split by newline or pipe
    if "\n" in text:
        return [x.strip() for x in text.split("\n") if x.strip()]
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]

    # Fallback: single item
    return [text]


def append_record_to_excel(record: dict, path: Path):
    """
    Append a single decision record to an Excel file.
    Creates the file if it doesn't exist.
    """
    df_new = pd.DataFrame([record])

    if path.exists():
        try:
            df_old = pd.read_excel(path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            # If the existing file is corrupted, just overwrite with new
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(path, index=False, engine="openpyxl")


@st.cache_resource
def get_pipeline_instance() -> Optional[UNSPSCPipeline]:
    """
    Create a single UNSPSCPipeline instance for the app lifetime.
    Assumes all config is handled inside unspsc_classifier_v10.py
    (you can adjust this if your pipeline needs arguments).
    """
    if not PIPELINE_AVAILABLE:
        return None

    # If your UNSPSCPipeline requires arguments, configure them here.
    # For now, assume default constructor works.
    try:
        pipeline = UNSPSCPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None


def classify_new_material(
    material_code: str,
    material_desc: str,
    po_text: str,
    challenge_level: str,
    item_detail: str,
) -> Optional[ClassificationResult]:
    """
    Run the UNSPSC pipeline for a single new material.
    Adjust arguments based on your actual UNSPSCPipeline API.
    """
    pipeline = get_pipeline_instance()
    if pipeline is None:
        st.error("Pipeline is not initialized; cannot classify new material.")
        return None

    try:
        # Adjust call signature if your pipeline expects different params
        result: ClassificationResult = pipeline.classify_material(
            material_code=material_code,
            material_desc=material_desc,
            po_text=po_text,
            challenge=challenge_level,
            item_detail=item_detail or "",
            matl_group="",
            plant="",
        )
        return result
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
        return None


# -------------------------------------------------------------------
# 4. STREAMLIT UI
# -------------------------------------------------------------------

st.set_page_config(
    page_title="UNSPSC Buyer Portal",
    page_icon="📦",
    layout="wide",
)

st.title("📦 UNSPSC Buyer Portal")

st.sidebar.header("Mode")

mode = st.sidebar.radio(
    "Select scenario",
    [
        "Scenario 1 – Buyer confirmation (existing materials)",
        "Scenario 2 – New material UNSPSC search",
    ],
)

with st.sidebar.expander("Your details"):
    buyer_name = st.text_input("Your name", value="")
    buyer_email = st.text_input("Your email", value="")

# Load existing classified results (for Scenario 1)
results_df = load_results_data()

if results_df is None:
    if mode.startswith("Scenario 1"):
        st.warning(
            f"`results.xlsx` not found in folder: {BASE_DIR}\n\n"
            "Upload or place `results.xlsx` next to unspsc_buyer_portal.py "
            "and redeploy the app."
        )

# -------------------------------------------------------------------
# 4A. SCENARIO 1 – BUYER CONFIRMATION
# -------------------------------------------------------------------

if mode.startswith("Scenario 1"):
    st.subheader("Scenario 1: Buyer confirmation of pipeline classification")

    col_search, col_result = st.columns([1, 2])

    with col_search:
        material_code_query = st.text_input("Material Code (exact match)")
        desc_query = st.text_input("Material Description (contains)")
        search_button = st.button("Search material")

    selected_row = None

    if search_button:
        if results_df is None or results_df.empty:
            st.error("No classified data available.")
        else:
            df = results_df.copy()

            if material_code_query:
                df = df[
                    df["Material"].astype(str)
                    == material_code_query.strip()
                ]

            if desc_query:
                df = df[
                    df["Material_Description"]
                    .astype(str)
                    .str.contains(desc_query, case=False, na=False)
                ]

            if df.empty:
                st.warning("No matching materials found.")
            else:
                st.success(f"Found {len(df)} matching material(s). Showing first one.")
                selected_row = df.iloc[0]

    if selected_row is not None:
        with col_result:
            st.markdown("### Material details")
            st.write(
                {
                    "Material": selected_row["Material"],
                    "Description": selected_row["Material_Description"],
                    "PO_Text": selected_row.get("PO_Text", ""),
                }
            )

            final_code = selected_row.get("Final_Recommendation")
            final_commodity = selected_row.get("Final_Commodity")
            final_conf = float(selected_row.get("Confidence", 0.0))

            st.markdown("### Pipeline final recommendation")
            st.metric(
                "Final UNSPSC",
                f"{final_code} – {final_commodity}",
                help=f"Confidence: {final_conf:.2f}",
            )
            st.caption(
                f"Decision rule: {selected_row.get('Decision_Rule', '')}"
            )

            # Top 5
            top5_list = parse_list_field(
                selected_row.get("Top_5_Recommendations", "")
            )
            st.markdown("#### Top 5 recommendations")
            if top5_list:
                for i, item in enumerate(top5_list, 1):
                    st.write(f"{i}. {item}")
            else:
                st.write("No Top 5 recommendations available.")

            # All ranked (lazy)
            all_ranked_list = parse_list_field(
                selected_row.get("All_Commodities_Ranked", "")
            )
            if all_ranked_list:
                show_all = st.checkbox("Show all ranked recommendations")
                if show_all:
                    st.markdown("#### All ranked recommendations")
                    for i, item in enumerate(all_ranked_list, 1):
                        st.write(f"{i}. {item}")

            st.markdown("---")
            st.markdown("### Buyer decision")

            decision = st.radio(
                "Your decision",
                [
                    "Accept final recommendation",
                    "Choose from Top 5",
                    "Choose from All ranked",
                    "No suitable match",
                ],
            )

            selected_code = None
            selected_commodity = None
            selected_source = None
            confidence = final_conf
            flagged = False

            if decision == "Accept final recommendation":
                selected_source = "final"
                selected_code = final_code
                selected_commodity = final_commodity

            elif decision == "Choose from Top 5":
                options = [""] + top5_list
                choice = st.selectbox("Select from Top 5", options)
                if choice:
                    selected_source = "top5"
                    selected_code = choice.split("-")[0].strip()
                    selected_commodity = "-".join(
                        choice.split("-")[1:]
                    ).strip()

            elif decision == "Choose from All ranked":
                options = [""] + all_ranked_list
                choice = st.selectbox("Select from All ranked", options)
                if choice:
                    selected_source = "all_ranked"
                    selected_code = choice.split("-")[0].strip()
                    selected_commodity = "-".join(
                        choice.split("-")[1:]
                    ).strip()

            elif decision == "No suitable match":
                selected_source = "no_match"
                selected_code = None
                selected_commodity = None
                confidence = 0.0
                flagged = True

            comments = st.text_area("Comments (optional)")

            if st.button("Save decision"):
                if not buyer_name or not buyer_email:
                    st.error("Please enter your name and email in the sidebar.")
                else:
                    record = {
                        "material_code": selected_row["Material"],
                        "material_description": selected_row[
                            "Material_Description"
                        ],
                        "selected_code": selected_code,
                        "selected_commodity": selected_commodity,
                        "selected_source": selected_source,
                        "confidence": confidence,
                        "buyer_name": buyer_name,
                        "buyer_email": buyer_email,
                        "decision": decision,
                        "comments": comments,
                        "created_at": datetime.utcnow(),
                    }
                    if flagged:
                        append_record_to_excel(record, FLAGGED_FILE)
                    else:
                        append_record_to_excel(record, CONFIRMED_FILE)

                    st.success(
                        "Decision saved successfully "
                        f"({'flagged' if flagged else 'confirmed'})."
                    )

# -------------------------------------------------------------------
# 4B. SCENARIO 2 – NEW MATERIAL UNSPSC SEARCH
# -------------------------------------------------------------------

else:
    st.subheader("Scenario 2: New material UNSPSC search")

    if not PIPELINE_AVAILABLE:
        st.error(
            "UNSPSC pipeline is not available. "
            "Ensure unspsc_classifier_v10.py is present and imports correctly."
        )
    else:
        with st.form("new_material_form"):
            col1, col2 = st.columns(2)
            with col1:
                material_code = st.text_input("Material Code")
                challenge_level = st.selectbox(
                    "Challenge Level", ["Low", "Medium", "High"], index=1
                )
            with col2:
                po_text = st.text_input("PO Text (optional)")
                item_detail = st.text_input("Item Detail (optional)")

            material_desc = st.text_area(
                "Material Description", height=120
            )

            submit = st.form_submit_button("Find UNSPSC")

        if submit:
            if not material_code or not material_desc:
                st.error("Material Code and Description are required.")
            else:
                with st.spinner("Running UNSPSC pipeline..."):
                    result = classify_new_material(
                        material_code=material_code,
                        material_desc=material_desc,
                        po_text=po_text,
                        challenge_level=challenge_level,
                        item_detail=item_detail,
                    )

                if result is None:
                    st.stop()

                st.success("Classification complete.")

                st.markdown("### Pipeline final recommendation")
                st.write(
                    {
                        "Material": result.material_code,
                        "Description": result.material_description,
                        "PO_Text": result.po_text,
                    }
                )

                st.metric(
                    "Final UNSPSC",
                    f"{result.final_recommendation} – {result.final_commodity}",
                    help=(
                        f"Confidence: {result.confidence:.2f} | "
                        f"Decision rule: {result.decision_rule}"
                    ),
                )

                st.markdown("#### Top 5 recommendations")
                for i, item in enumerate(
                    result.top_5_recommendations or [], 1
                ):
                    st.write(f"{i}. {item}")

                if result.all_commodities_ranked:
                    if st.checkbox(
                        "Show all ranked recommendations (new material)"
                    ):
                        st.markdown("#### All ranked recommendations")
                        for i, item in enumerate(
                            result.all_commodities_ranked, 1
                        ):
                            st.write(f"{i}. {item}")

                st.markdown("---")
                st.markdown("### Buyer decision for this new material")

                decision = st.radio(
                    "Your decision",
                    [
                        "Accept final recommendation",
                        "Choose from Top 5",
                        "Choose from All ranked",
                        "No suitable match",
                    ],
                    key="new_decision",
                )

                selected_code = None
                selected_commodity = None
                selected_source = None
                confidence = result.confidence
                flagged = False

                if decision == "Accept final recommendation":
                    selected_source = "final"
                    selected_code = result.final_recommendation
                    selected_commodity = result.final_commodity

                elif decision == "Choose from Top 5":
                    options = [""] + (result.top_5_recommendations or [])
                    choice = st.selectbox(
                        "Select from Top 5",
                        options,
                        key="new_top5_choice",
                    )
                    if choice:
                        selected_source = "top5"
                        selected_code = choice.split("-")[0].strip()
                        selected_commodity = "-".join(
                            choice.split("-")[1:]
                        ).strip()

                elif decision == "Choose from All ranked":
                    options = [""] + (result.all_commodities_ranked or [])
                    choice = st.selectbox(
                        "Select from All ranked",
                        options,
                        key="new_all_choice",
                    )
                    if choice:
                        selected_source = "all_ranked"
                        selected_code = choice.split("-")[0].strip()
                        selected_commodity = "-".join(
                            choice.split("-")[1:]
                        ).strip()

                elif decision == "No suitable match":
                    selected_source = "no_match"
                    selected_code = None
                    selected_commodity = None
                    confidence = 0.0
                    flagged = True

                comments = st.text_area(
                    "Comments (optional)", key="new_comments"
                )

                if st.button("Save decision for new material"):
                    if not buyer_name or not buyer_email:
                        st.error(
                            "Please enter your name and email in the sidebar."
                        )
                    else:
                        record = {
                            "material_code": material_code,
                            "material_description": material_desc,
                            "selected_code": selected_code,
                            "selected_commodity": selected_commodity,
                            "selected_source": selected_source,
                            "confidence": confidence,
                            "buyer_name": buyer_name,
                            "buyer_email": buyer_email,
                            "decision": decision,
                            "comments": comments,
                            "created_at": datetime.utcnow(),
                        }
                        if flagged:
                            append_record_to_excel(record, FLAGGED_FILE)
                        else:
                            append_record_to_excel(record, CONFIRMED_FILE)

                        st.success(
                            "Decision saved successfully "
                            f"({'flagged' if flagged else 'confirmed'})."
                        )
