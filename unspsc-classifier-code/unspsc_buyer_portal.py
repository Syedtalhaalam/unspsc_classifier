#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNSPSC Buyer Confirmation Portal - Enterprise Grade
===================================================

A comprehensive Streamlit web portal for UNSPSC classification workflow:
- Scenario 1: Buyer confirmation of pipeline classifications
- Scenario 2: Real-time classification for new materials

Author: UNSPSC Classification Team
Version: 1.0
Last Updated: 2025-01-16
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import json

# Add the directory containing the pipeline to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the pipeline - will be imported dynamically
try:
    from unspsc_classifier_v10 import UNSPSCPipeline, setup_logging, ClassificationResult
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logging.warning("Pipeline not available - real-time classification disabled")

# =============================================================================
# Configuration
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="UNSPSC Buyer Portal",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("confirmed_results")
RESULTS_FILE = DATA_DIR / "results.xlsx"
CONFIRMED_FILE = OUTPUT_DIR / "confirmed_materials.xlsx"
FLAGGED_FILE = OUTPUT_DIR / "flagged_materials.xlsx"
UNSPSC_CATALOG = DATA_DIR / "unspsc_catalog.csv"
CACHE_DIR = Path("cache")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# Custom CSS for Professional Look
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1f4788;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1f4788;
    }
    
    h2 {
        color: #2563eb;
        margin-top: 2rem;
    }
    
    h3 {
        color: #3b82f6;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
        border: 2px solid #2563eb;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    
    /* Material card styling */
    .material-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .material-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f4788;
        margin-bottom: 0.5rem;
    }
    
    .material-detail {
        color: #6b7280;
        margin: 0.25rem 0;
    }
    
    .code-box {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    
    .code-title {
        font-weight: 700;
        color: #1f4788;
        font-size: 1.1rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .confidence-high {
        background: #10b981;
        color: white;
    }
    
    .confidence-medium {
        background: #f59e0b;
        color: white;
    }
    
    .confidence-low {
        background: #ef4444;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .status-confirmed {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-pending {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-flagged {
        color: #ef4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Utility Functions
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'confirmed_count' not in st.session_state:
        st.session_state.confirmed_count = 0
    if 'flagged_count' not in st.session_state:
        st.session_state.flagged_count = 0
    if 'current_material' not in st.session_state:
        st.session_state.current_material = None
    if 'show_top5' not in st.session_state:
        st.session_state.show_top5 = False
    if 'show_all' not in st.session_state:
        st.session_state.show_all = False
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None

def load_results_data() -> Optional[pd.DataFrame]:
    """Load the classification results Excel file"""
    try:
        if RESULTS_FILE.exists():
            df = pd.read_excel(RESULTS_FILE)
            return df
        else:
            st.error(f"⚠️ Results file not found: {RESULTS_FILE}")
            st.info("Please ensure the classification results file is placed in the 'data' folder.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading results: {str(e)}")
        return None

def load_confirmed_materials() -> pd.DataFrame:
    """Load already confirmed materials"""
    try:
        if CONFIRMED_FILE.exists():
            return pd.read_excel(CONFIRMED_FILE)
        else:
            # Create empty dataframe with expected structure
            return pd.DataFrame(columns=[
                'Date_Confirmed', 'Buyer_Name', 'Material_Code', 'Material_Description',
                'Final_Recommendation', 'Commodity_Code', 'Segment', 'Family', 'Class',
                'Confirmed_Code', 'Confirmed_Title', 'Confidence', 'Buyer_Comments'
            ])
    except Exception as e:
        logging.error(f"Error loading confirmed materials: {e}")
        return pd.DataFrame()

def load_flagged_materials() -> pd.DataFrame:
    """Load flagged materials"""
    try:
        if FLAGGED_FILE.exists():
            return pd.read_excel(FLAGGED_FILE)
        else:
            return pd.DataFrame(columns=[
                'Date_Flagged', 'Buyer_Name', 'Material_Code', 'Material_Description',
                'Final_Recommendation', 'Flag_Reason', 'Buyer_Comments'
            ])
    except Exception as e:
        logging.error(f"Error loading flagged materials: {e}")
        return pd.DataFrame()

def save_confirmed_material(material_data: Dict, confirmed_code: str, confirmed_title: str, 
                           buyer_name: str, comments: str = ""):
    """Save confirmed material to Excel"""
    try:
        confirmed_df = load_confirmed_materials()
        
        new_entry = {
            'Date_Confirmed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Buyer_Name': buyer_name,
            'Material_Code': material_data.get('Material_Code', ''),
            'Material_Description': material_data.get('Material_Description', ''),
            'Final_Recommendation': material_data.get('Final_Recommendation', ''),
            'Commodity_Code': material_data.get('Commodity_Code', ''),
            'Segment': material_data.get('Segment', ''),
            'Family': material_data.get('Family', ''),
            'Class': material_data.get('Class', ''),
            'Confirmed_Code': confirmed_code,
            'Confirmed_Title': confirmed_title,
            'Confidence': material_data.get('Confidence', 0),
            'Buyer_Comments': comments
        }
        
        confirmed_df = pd.concat([confirmed_df, pd.DataFrame([new_entry])], ignore_index=True)
        confirmed_df.to_excel(CONFIRMED_FILE, index=False, engine='openpyxl')
        
        st.session_state.confirmed_count += 1
        return True
    except Exception as e:
        st.error(f"❌ Error saving confirmed material: {str(e)}")
        return False

def save_flagged_material(material_data: Dict, flag_reason: str, 
                         buyer_name: str, comments: str = ""):
    """Save flagged material to Excel"""
    try:
        flagged_df = load_flagged_materials()
        
        new_entry = {
            'Date_Flagged': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Buyer_Name': buyer_name,
            'Material_Code': material_data.get('Material_Code', ''),
            'Material_Description': material_data.get('Material_Description', ''),
            'Final_Recommendation': material_data.get('Final_Recommendation', ''),
            'Flag_Reason': flag_reason,
            'Buyer_Comments': comments
        }
        
        flagged_df = pd.concat([flagged_df, pd.DataFrame([new_entry])], ignore_index=True)
        flagged_df.to_excel(FLAGGED_FILE, index=False, engine='openpyxl')
        
        st.session_state.flagged_count += 1
        return True
    except Exception as e:
        st.error(f"❌ Error saving flagged material: {str(e)}")
        return False

def parse_recommendations(recommendations_str: str) -> List[Tuple[str, str]]:
    """Parse recommendations string into list of (code, title) tuples"""
    if pd.isna(recommendations_str) or not recommendations_str:
        return []
    
    recommendations = []
    lines = str(recommendations_str).split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse "CODE - TITLE" format
        if ' - ' in line:
            parts = line.split(' - ', 1)
            code = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else ""
            
            # Remove confidence scores if present
            if '(confidence:' in title:
                title = title.split('(confidence:')[0].strip()
            
            recommendations.append((code, title))
    
    return recommendations

def search_material(results_df: pd.DataFrame, search_query: str) -> Optional[pd.Series]:
    """Search for material by code or description"""
    search_query = search_query.strip()
    
    # Try exact match on material code
    if search_query.isdigit():
        match = results_df[results_df['Material_Code'].astype(str) == search_query]
        if not match.empty:
            return match.iloc[0]
    
    # Try partial match on description
    match = results_df[
        results_df['Material_Description'].str.contains(search_query, case=False, na=False)
    ]
    if not match.empty:
        return match.iloc[0]
    
    # Try match on material code as string
    match = results_df[
        results_df['Material_Code'].astype(str).str.contains(search_query, case=False, na=False)
    ]
    if not match.empty:
        return match.iloc[0]
    
    return None

def format_confidence(confidence: float) -> str:
    """Format confidence score with color coding"""
    if confidence >= 0.9:
        badge_class = "confidence-high"
        label = "High"
    elif confidence >= 0.7:
        badge_class = "confidence-medium"
        label = "Medium"
    else:
        badge_class = "confidence-low"
        label = "Low"
    
    return f'<span class="confidence-badge {badge_class}">{label} ({confidence:.0%})</span>'

# =============================================================================
# Real-time Classification Functions (Scenario 2)
# =============================================================================

def run_pipeline_classification(material_code: str, material_description: str, 
                                po_text: str = "", item_detail: str = "") -> Optional[ClassificationResult]:
    """Run the classification pipeline for a new material"""
    if not PIPELINE_AVAILABLE:
        st.error("❌ Classification pipeline is not available. Please check installation.")
        return None
    
    try:
        with st.spinner("🔄 Running classification pipeline... This may take 30-60 seconds..."):
            # Initialize pipeline
            pipeline = UNSPSCPipeline(
                unspsc_csv_path=str(UNSPSC_CATALOG),
                cache_file=str(CACHE_DIR / "cache.jsonl")
            )
            
            # Run classification
            result = pipeline.classify_material(
                material_code=material_code,
                material_description=material_description,
                po_text=po_text,
                challenge_level="Medium",
                item_detail=item_detail,
                matl_group="",
                plant=""
            )
            
            return result
            
    except Exception as e:
        st.error(f"❌ Classification failed: {str(e)}")
        logging.error(f"Pipeline classification error: {e}", exc_info=True)
        return None

def display_classification_result(result: ClassificationResult):
    """Display classification results in a nice format"""
    st.markdown("### 🎯 Classification Results")
    
    # Main recommendation
    st.markdown(f"""
    <div class="code-box">
        <div class="code-title">📌 Final Recommendation</div>
        <div style="font-size: 1.2rem; margin-top: 0.5rem;">
            <strong>{result.commodity_code}</strong> - {result.final_recommendation}
        </div>
        <div style="margin-top: 0.5rem;">
            {format_confidence(result.confidence)}
            <span style="color: #6b7280; margin-left: 1rem;">
                {result.decision_rule.replace('_', ' ').title()}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hierarchy
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Segment", result.segment or "N/A")
    with col2:
        st.metric("Family", result.family or "N/A")
    with col3:
        st.metric("Class", result.class_name or "N/A")
    with col4:
        st.metric("Processing Time", f"{result.processing_time:.1f}s")
    
    # Individual LLM results
    with st.expander("🤖 Individual LLM Classifications"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ChatGPT**")
            st.write(result.gpt_commodity or "N/A")
            
        with col2:
            st.markdown("**Gemini**")
            st.write(result.gemini_commodity or "N/A")
            
        with col3:
            st.markdown("**Claude**")
            st.write(result.claude_commodity or "N/A")
    
    # Top recommendations
    if result.top_5_recommendations:
        with st.expander("📊 Top 5 Recommendations"):
            recs = parse_recommendations(result.top_5_recommendations)
            for i, (code, title) in enumerate(recs[:5], 1):
                st.write(f"{i}. **{code}** - {title}")
    
    # All ranked commodities
    if result.all_30_ranked:
        with st.expander("📋 All Commodities Ranked"):
            st.text_area("All Rankings", result.all_30_ranked, height=200)

# =============================================================================
# Main Application UI
# =============================================================================

def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🏢 UNSPSC Buyer Confirmation Portal")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f4788/ffffff?text=K-Electric", 
                use_column_width=True)
        st.markdown("### 👤 Buyer Information")
        
        buyer_name = st.text_input("Your Name", value="", key="buyer_name")
        
        if not buyer_name:
            st.warning("⚠️ Please enter your name to continue")
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        # Load statistics
        confirmed_df = load_confirmed_materials()
        flagged_df = load_flagged_materials()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <div class="metric-value">{len(confirmed_df)}</div>
                <div class="metric-label">Confirmed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <div class="metric-value">{len(flagged_df)}</div>
                <div class="metric-label">Flagged</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ Help")
        with st.expander("📖 How to Use"):
            st.markdown("""
            **Scenario 1: Confirm Existing Classifications**
            1. Enter material code or description
            2. Review the recommended UNSPSC code
            3. Confirm if correct, or view alternatives
            4. Add comments if needed
            
            **Scenario 2: Classify New Materials**
            1. Switch to "New Material" tab
            2. Enter material details
            3. Click "Find UNSPSC Code"
            4. Review and confirm results
            """)
    
    # Main content
    if not buyer_name:
        st.info("👈 Please enter your name in the sidebar to begin")
        return
    
    # Tabs for two scenarios
    tab1, tab2 = st.tabs(["📋 Confirm Existing Materials", "🆕 Classify New Material"])
    
    # =============================================================================
    # Scenario 1: Buyer Confirmation
    # =============================================================================
    
    with tab1:
        st.markdown("## Scenario 1: Confirm Pipeline Classifications")
        st.markdown("Search for materials and confirm their UNSPSC codes")
        
        # Load results
        results_df = load_results_data()
        
        if results_df is None or results_df.empty:
            st.warning("⚠️ No classification results found. Please ensure results.xlsx is in the data folder.")
            return
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search Material",
                placeholder="Enter material code or description...",
                key="search_query"
            )
        
        with col2:
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_button and search_query:
            material = search_material(results_df, search_query)
            
            if material is not None:
                st.session_state.current_material = material.to_dict()
                st.session_state.show_top5 = False
                st.session_state.show_all = False
                st.success(f"✅ Found material: {material['Material_Code']}")
            else:
                st.error(f"❌ No material found matching '{search_query}'")
                st.info("💡 Tip: Try searching by material code or a few keywords from the description")
        
        # Display current material
        if st.session_state.current_material:
            material = st.session_state.current_material
            
            st.markdown("---")
            
            # Material details card
            st.markdown(f"""
            <div class="material-card">
                <div class="material-header">
                    📦 Material: {material['Material_Code']}
                </div>
                <div class="material-detail">
                    <strong>Description:</strong> {material['Material_Description']}
                </div>
                {f'<div class="material-detail"><strong>Item Detail:</strong> {material["Item_Detail"]}</div>' if pd.notna(material.get('Item_Detail')) else ''}
                {f'<div class="material-detail"><strong>Material Group:</strong> {material["Matl_Group"]}</div>' if pd.notna(material.get('Matl_Group')) else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Final recommendation
            st.markdown("### 🎯 Pipeline Recommendation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="code-box">
                    <div class="code-title">Final UNSPSC Code</div>
                    <div style="font-size: 1.3rem; margin: 0.75rem 0;">
                        <strong>{material['Commodity_Code']}</strong> - {material['Final_Recommendation']}
                    </div>
                    <div>
                        <strong>Segment:</strong> {material.get('Segment', 'N/A')}<br>
                        <strong>Family:</strong> {material.get('Family', 'N/A')}<br>
                        <strong>Class:</strong> {material.get('Class', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Classification Details**")
                st.write(f"**Confidence:** {format_confidence(material.get('Confidence', 0))}", unsafe_allow_html=True)
                st.write(f"**Decision Rule:** {material.get('Decision_Rule', 'N/A').replace('_', ' ').title()}")
                st.write(f"**Processing Time:** {material.get('Processing_Time', 0):.1f}s")
            
            # Action buttons
            st.markdown("### ✅ Your Decision")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Confirm This Code", type="primary", use_container_width=True, 
                           key="confirm_main"):
                    comments = st.session_state.get('buyer_comments', '')
                    if save_confirmed_material(
                        material,
                        str(material['Commodity_Code']),
                        material['Final_Recommendation'],
                        buyer_name,
                        comments
                    ):
                        st.success(f"✅ Material {material['Material_Code']} confirmed!")
                        st.balloons()
                        st.session_state.current_material = None
                        st.rerun()
            
            with col2:
                if not st.session_state.show_top5:
                    if st.button("🔍 View More Options", use_container_width=True, key="show_top5"):
                        st.session_state.show_top5 = True
                        st.rerun()
            
            with col3:
                flag_options = [
                    "Select reason...",
                    "Incorrect classification",
                    "Missing details",
                    "Ambiguous description",
                    "No suitable match",
                    "Other"
                ]
                flag_reason = st.selectbox("⚠️ Flag Material", flag_options, key="flag_reason")
                
                if flag_reason != "Select reason...":
                    if st.button("⚠️ Flag This Material", use_container_width=True, key="flag_material"):
                        comments = st.session_state.get('buyer_comments', '')
                        if save_flagged_material(material, flag_reason, buyer_name, comments):
                            st.warning(f"⚠️ Material {material['Material_Code']} flagged for review")
                            st.session_state.current_material = None
                            st.rerun()
            
            # Comments section
            st.markdown("### 💬 Comments (Optional)")
            st.text_area(
                "Add any notes or feedback",
                key="buyer_comments",
                placeholder="Example: Classification looks correct, material description could be more specific...",
                height=100
            )
            
            # Show Top 5 Recommendations
            if st.session_state.show_top5:
                st.markdown("---")
                st.markdown("### 📊 Top 5 Alternative Recommendations")
                
                top5 = parse_recommendations(material.get('Top_5_Recommendations', ''))
                
                if top5:
                    for i, (code, title) in enumerate(top5, 1):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style="padding: 0.75rem; background: #f9fafb; border-radius: 0.5rem; margin: 0.5rem 0;">
                                <strong>{i}. {code}</strong> - {title}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button(f"✅ Select", key=f"select_top5_{i}", use_container_width=True):
                                comments = st.session_state.get('buyer_comments', '')
                                if save_confirmed_material(material, code, title, buyer_name, comments):
                                    st.success(f"✅ Selected alternative: {code}")
                                    st.session_state.current_material = None
                                    st.session_state.show_top5 = False
                                    st.rerun()
                    
                    # Button to show all commodities
                    if not st.session_state.show_all:
                        if st.button("🔍 View All Ranked Commodities", key="show_all_btn"):
                            st.session_state.show_all = True
                            st.rerun()
                else:
                    st.info("No alternative recommendations available")
            
            # Show All Ranked Commodities
            if st.session_state.show_all:
                st.markdown("---")
                st.markdown("### 📋 All Ranked Commodities")
                
                all_ranked = parse_recommendations(material.get('All_Commodities_Ranked', ''))
                
                if all_ranked:
                    # Display in expandable sections of 10
                    for start_idx in range(0, len(all_ranked), 10):
                        end_idx = min(start_idx + 10, len(all_ranked))
                        with st.expander(f"Commodities {start_idx + 1} - {end_idx}"):
                            for i, (code, title) in enumerate(all_ranked[start_idx:end_idx], start_idx + 1):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"{i}. **{code}** - {title}")
                                
                                with col2:
                                    if st.button(f"✅", key=f"select_all_{i}", use_container_width=True):
                                        comments = st.session_state.get('buyer_comments', '')
                                        if save_confirmed_material(material, code, title, buyer_name, comments):
                                            st.success(f"✅ Selected: {code}")
                                            st.session_state.current_material = None
                                            st.session_state.show_all = False
                                            st.session_state.show_top5 = False
                                            st.rerun()
                else:
                    st.info("No ranked commodities available")
    
    # =============================================================================
    # Scenario 2: Real-time Classification
    # =============================================================================
    
    with tab2:
        st.markdown("## Scenario 2: Classify New Material")
        st.markdown("Enter details for a material not in the current classification results")
        
        if not PIPELINE_AVAILABLE:
            st.error("❌ Classification pipeline is not available. Please check the installation guide.")
            st.info("Ensure unspsc_classifier_v10.py is in the same directory as this portal.")
            return
        
        # Input form
        with st.form("new_material_form"):
            st.markdown("### 📝 Material Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_material_code = st.text_input(
                    "Material Code *",
                    placeholder="e.g., 10000355",
                    help="Enter the unique material code"
                )
                
                new_material_desc = st.text_area(
                    "Material Description *",
                    placeholder="e.g., BASE; FUSE HRC LT 630 A (NH3)",
                    help="Enter the full material description",
                    height=100
                )
            
            with col2:
                new_po_text = st.text_area(
                    "PO Text (Optional)",
                    placeholder="Additional purchase order text...",
                    help="Any additional context from purchase orders",
                    height=100
                )
                
                new_item_detail = st.text_area(
                    "Item Detail (Optional)",
                    placeholder="Detailed specifications or usage context...",
                    help="Detailed information about the material's function and usage",
                    height=100
                )
            
            submit_button = st.form_submit_button("🔍 Find UNSPSC Code", type="primary", use_container_width=True)
        
        # Process classification
        if submit_button:
            if not new_material_code or not new_material_desc:
                st.error("❌ Please provide both Material Code and Material Description")
            else:
                # Check if material already exists
                results_df = load_results_data()
                if results_df is not None:
                    existing = search_material(results_df, new_material_code)
                    if existing is not None:
                        st.warning("⚠️ This material already exists in the classification results!")
                        st.info("💡 Please use Scenario 1 (Confirm Existing Materials) to review this material")
                    else:
                        # Run classification
                        result = run_pipeline_classification(
                            new_material_code,
                            new_material_desc,
                            new_po_text,
                            new_item_detail
                        )
                        
                        if result:
                            st.session_state.classification_result = result
        
        # Display classification result
        if st.session_state.classification_result:
            result = st.session_state.classification_result
            
            st.markdown("---")
            display_classification_result(result)
            
            # Confirmation section
            st.markdown("---")
            st.markdown("### ✅ Confirm Classification")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_comments = st.text_area(
                    "Comments (Optional)",
                    key="new_material_comments",
                    placeholder="Add any notes about this classification...",
                    height=100
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ Confirm & Save", type="primary", use_container_width=True, key="confirm_new"):
                    # Prepare material data
                    material_data = {
                        'Material_Code': new_material_code,
                        'Material_Description': new_material_desc,
                        'Final_Recommendation': result.final_recommendation,
                        'Commodity_Code': result.commodity_code,
                        'Segment': result.segment,
                        'Family': result.family,
                        'Class': result.class_name,
                        'Confidence': result.confidence
                    }
                    
                    if save_confirmed_material(
                        material_data,
                        str(result.commodity_code),
                        result.final_recommendation,
                        buyer_name,
                        new_comments
                    ):
                        st.success(f"✅ New material {new_material_code} classified and confirmed!")
                        st.balloons()
                        st.session_state.classification_result = None
                        st.rerun()
                
                if st.button("❌ Cancel", use_container_width=True, key="cancel_new"):
                    st.session_state.classification_result = None
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p>UNSPSC Buyer Portal v1.0 | K-Electric Procurement Team</p>
        <p style="font-size: 0.875rem;">For support, contact: procurement.support@ke.com.pk</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
