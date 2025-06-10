import streamlit as st
from dataclasses import asdict
import tempfile
import os
import json
import pandas as pd
from datetime import datetime
import logging
import traceback

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ultimate Term Corrector V8", page_icon="🚀", layout="wide")

# --- IMPORT BACKEND ---
try:
    from ultimate_term_corrector import UltimateTermCorrectorV8, TermCorrection
    CORRECTOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Fatal Error: Could not import backend. Ensure 'ultimate_term_corrector.py' is in the same folder. Details: {e}")
    CORRECTOR_AVAILABLE = False
    st.stop()

# --- HELPER FUNCTIONS ---

def initialize_session_state():
    """Initializes session state variables."""
    defaults = {
        'terms': [], 'corrector': None, 'uploaded_file_info': None,
        'processing_results': None, 'force_mode': False, 'logger': None,
        'detected_source_lang': 'en', 'detected_target_lang': 'tr'
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def process_file_with_terms():
    """Main function to orchestrate the file processing with real-time progress."""
    progress_container = st.container()
    with progress_container:
        st.info("🔄 Processing started...")
        status_text = st.empty()
        progress_bar = st.progress(0, text="Initializing...")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_info['name'])[-1]) as tmp_file:
            tmp_file.write(st.session_state.uploaded_file_info['bytes'])
            tmp_path = tmp_file.name

        corrector = st.session_state.corrector
        corrector.term_corrections = [TermCorrection(**term_data) for term_data in st.session_state.terms]

        total_terms = len(corrector.term_corrections)
        total_batches = 0  # We will know this after preprocessing

        # Define callbacks for real-time progress updates
        def variant_progress_callback(current, total):
            progress = int((current / total) * 20)  # Variant generation takes up first 20%
            progress_bar.progress(progress, text=f"🧠 Step 1/2: Generating term variants ({current}/{total})...")

        def batch_progress_callback(current, total):
            # This needs total_batches which we don't know yet. We'll handle this inside the main logic
            progress = 20 + int((current / total) * 80) # Main processing takes up the other 80%
            progress_bar.progress(progress, text=f"⚙️ Step 2/2: Processing file batches ({current}/{total})...")

        # Create a dictionary of callbacks to pass to the backend
        callbacks = {
            'variants': variant_progress_callback,
            'batches': batch_progress_callback
        }
        
        # Run the main processing method from the backend, passing the callbacks
        corrections_made, detailed_results = corrector.process_file_v8(tmp_path, st.session_state.logger, progress_callbacks=callbacks)
        
        progress_bar.progress(100, text="✅ Process complete!")
        
        corrected_content = ""
        if corrections_made > 0:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                corrected_content = f.read()
        
        st.session_state.processing_results = {
            'corrections_made': corrections_made, 'detailed_results': detailed_results,
            'corrected_content': corrected_content, 'stats': corrector.processing_stats
        }
        
        st.success(f"🎉 Process finished! {corrections_made} corrections were applied.")
        st.info("📊 View the 'Results' tab for details and downloads.")
        
    except Exception as e:
        st.error(f"❌ A critical error occurred during processing: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- MAIN APP LAYOUT ---

def main():
    st.title("🚀 Ultimate Term Corrector V8")
    st.markdown("#### AI-Powered Terminology Correction with Selectable Processing Modes")
    
    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success("✅ Backend module loaded!")
        
        api_key = st.text_input("🔑 Claude API Key", type="password", help="Required to run the corrector.")
        
        mode_choice = st.radio(
            "⚙️ Processing Mode", ('AI-Evaluated', 'Forced Replacement'),
            help="**AI-Evaluated:** Corrects terms only where necessary. **Forced Replacement:** Replaces every identified instance."
        )
        st.session_state.force_mode = (mode_choice == 'Forced Replacement')

        if api_key:
            if st.session_state.corrector is None or st.session_state.corrector.force_mode != st.session_state.force_mode:
                try:
                    st.session_state.corrector = UltimateTermCorrectorV8(api_key, force_mode=st.session_state.force_mode)
                    st.session_state.logger = st.session_state.corrector.setup_logging()
                    st.success("✅ Corrector initialized!")
                except Exception as e:
                    st.error(f"❌ Corrector could not be initialized: {e}")
        else:
            st.warning("Please enter your API key.")

    if not st.session_state.corrector:
        st.info("⚠️ Enter your API key in the sidebar to begin.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📁 1. File Upload", "🔤 2. Term Management", "🚀 3. Process", "📊 4. Results"])

    with tab1:
        st.header("📁 File Upload & Analysis")
        uploaded_file = st.file_uploader("Select a bilingual file", type=['xliff', 'xlf', 'xml', 'sdlxliff', 'mqxliff'])
        
        if uploaded_file:
            st.session_state.uploaded_file_info = {'name': uploaded_file.name, 'bytes': uploaded_file.getvalue()}
            st.success(f"✅ Received: **{uploaded_file.name}**")
            
            with st.spinner("Analyzing file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    corrector = st.session_state.corrector
                    format_info = corrector.format_detector.detect_format(tmp_path)
                    # **UPDATE 1: Store detected languages in session state**
                    source_lang, target_lang = corrector.detect_languages_with_fallback(tmp_path, st.session_state.logger)
                    st.session_state.detected_source_lang = source_lang
                    st.session_state.detected_target_lang = target_lang
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Detected Format", format_info.get('type', 'N/A').upper())
                    col2.metric("Strategy", format_info.get('strategy', 'N/A').replace('_', ' ').title())
                    if source_lang and target_lang:
                        st.info(f"🌐 **Detected Languages:** {source_lang.upper()} ➡️ {target_lang.upper()} (Now set in Tab 2)")
                finally:
                    os.unlink(tmp_path)

    with tab2:
        st.header("🔤 Term Management")
        # **UPDATE 1: Use detected languages as default values**
        col1, col2 = st.columns(2)
        source_lang = col1.text_input("Source Language", value=st.session_state.get('detected_source_lang', 'en')).lower()
        target_lang = col2.text_input("Target Language", value=st.session_state.get('detected_target_lang', 'tr')).lower()
        
        with st.form("add_term_form", clear_on_submit=True):
            st.subheader("➕ Add New Term")
            c1, c2 = st.columns(2)
            source_term = c1.text_input(f"🔍 Source Term ({source_lang.upper()})")
            target_term = c2.text_input(f"✏️ Target Term ({target_lang.upper()})")
            description = st.text_input("📋 Description (Optional)")
            if st.form_submit_button("➕ Add Term", use_container_width=True):
                if source_term and target_term:
                    st.session_state.terms.append({'source_term': source_term, 'target_term': target_term, 'source_language': source_lang, 'target_language': target_lang, 'description': description, 'term_id': len(st.session_state.terms) + 1})
                    st.success(f"Added: '{source_term}' → '{target_term}'")
                else: st.error("Source and target terms are required.")

        if st.session_state.terms:
            st.subheader(f"📋 Term List ({len(st.session_state.terms)} terms)")
            df = pd.DataFrame(st.session_state.terms)
            st.dataframe(df[['term_id', 'source_term', 'target_term', 'description']], use_container_width=True)
            if st.button("🗑️ Clear All Terms", type="secondary"):
                st.session_state.terms = []
                st.rerun()

    with tab3:
        st.header("🚀 Process File")
        prereqs = {"API Key Entered": st.session_state.corrector is not None, "File Uploaded": st.session_state.uploaded_file_info is not None, "Terms Added": len(st.session_state.terms) > 0}
        st.subheader("📋 Pre-flight Check")
        for item, status in prereqs.items(): st.markdown(f"- {item}: {'✅' if status else '❌'}")
        
        if all(prereqs.values()):
            st.success("Ready to process!")
            st.markdown("---"); st.subheader("⚙️ Processing Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("📄 File Name", st.session_state.uploaded_file_info['name'])
            col2.metric("🔤 Term Count", len(st.session_state.terms))
            col3.metric("⚙️ Mode", "Forced" if st.session_state.force_mode else "AI-Evaluated")
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                process_file_with_terms()
        else: st.warning("Please complete all steps in previous tabs.")

    with tab4:
        st.header("📊 Results")
        if st.session_state.processing_results:
            res, stats = st.session_state.processing_results, st.session_state.processing_results.get('stats', {})
            st.subheader("📈 Summary Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔧 Corrections Made", stats.get('corrections_made', 0))
            c2.metric("📊 Units Processed", stats.get('units_processed', 0))
            c3.metric("⚡ Performance Gain", f"{stats.get('performance_gain', 1.0):.1f}x")
            c4.metric("💾 Cache Hits", stats.get('cache_hits', 0))
            
            st.markdown("---"); st.subheader("📥 Downloads")
            c1, c2 = st.columns(2)
            if res.get('corrected_content'):
                name, ext = os.path.splitext(st.session_state.uploaded_file_info['name'])
                c1.download_button("📥 Download Corrected File", res['corrected_content'].encode('utf-8'), f"{name}_corrected{ext}", "application/xml", use_container_width=True)
            report_json = json.dumps(res, indent=2, ensure_ascii=False, default=str)
            c2.download_button("📊 Download JSON Report", report_json, f"report_{datetime.now().strftime('%Y%m%d')}.json", "application/json", use_container_width=True)
            
            if res.get('detailed_results'):
                st.markdown("---"); st.subheader("📝 Detailed Changes")
                changes = [asdict(r) for r in res['detailed_results'] if r.new_target != r.original_target]
                if changes: st.dataframe(pd.DataFrame(changes)[['unit_id', 'original_target', 'new_target', 'applied_corrections']], use_container_width=True)
                else: st.info("No changes were applied to the file content.")
        else:
            st.info("Process a file to see the results here.")

if __name__ == "__main__":
    main()
