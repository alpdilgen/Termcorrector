#!/usr/bin/env python3
"""
Universal Term Corrector - Streamlit Web Application (Minimal Version)
====================================================
"""

import streamlit as st
import tempfile
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import the corrector - DÜZELTME: Dosya ismini doğru kullanın
try:
    from universal_term_corrector import UniversalTermCorrectorForce, TermCorrection, FileFormatInfo
except ImportError:
    st.error("❌ universal_term_corrector.py dosyası bulunamadı. Dosyanın aynı klasörde olduğundan emin olun.")
    st.info("💡 Orijinal dosyanızı 'universal_term_corrector.py' olarak yeniden adlandırın.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Universal Term Corrector",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'corrector' not in st.session_state:
        st.session_state.corrector = None
    if 'terms' not in st.session_state:
        st.session_state.terms = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'format_info' not in st.session_state:
        st.session_state.format_info = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

def create_corrector(api_key: str):
    """Create and cache the corrector instance"""
    try:
        corrector = UniversalTermCorrectorForce(api_key)
        st.session_state.corrector = corrector
        return corrector
    except Exception as e:
        st.error(f"❌ Corrector başlatılırken hata: {str(e)}")
        st.info("💡 API anahtarınızın geçerli olduğundan emin olun.")
        return None

def detect_file_format(uploaded_file):
    """Detect and display file format information"""
    if uploaded_file is None:
        return None
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        corrector = st.session_state.corrector
        if corrector:
            format_info = corrector.detect_bilingual_format(tmp_path)
            
            # Display format information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="feature-box">
                    <h4>📋 Dosya Formatı</h4>
                    <p><strong>{format_info.format_type.upper()}</strong> v{format_info.version}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="feature-box">
                    <h4>🔧 Yapı Tipi</h4>
                    <p>{format_info.structure_type.replace('_', ' ').title()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="feature-box">
                    <h4>✨ Özellikler</h4>
                    <p>{len(format_info.special_features)} özel özellik</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show detailed features
            if format_info.special_features:
                st.markdown("**Özel Özellikler:**")
                for feature in format_info.special_features:
                    st.markdown(f"• {feature.replace('_', ' ').title()}")
            
            return format_info, tmp_path
        
    except Exception as e:
        st.error(f"❌ Format tespitinde hata: {str(e)}")
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return None, None

def process_file():
    """Process the uploaded file with term corrections"""
    if not st.session_state.corrector:
        st.error("❌ Lütfen önce API anahtarınızı girin")
        return
    
    if not st.session_state.uploaded_file:
        st.error("❌ Lütfen önce bir dosya yükleyin")
        return
    
    if not st.session_state.terms:
        st.error("❌ Lütfen en az bir terim düzeltmesi ekleyin")
        return
    
    # Show processing status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(st.session_state.uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        status_text.text("🔍 Dosya formatı analiz ediliyor...")
        progress_bar.progress(10)
        
        corrector = st.session_state.corrector
        
        # Setup term corrections
        corrector.term_corrections = []
        for i, term_data in enumerate(st.session_state.terms):
            term_correction = TermCorrection(
                source_term=term_data['source_term'],
                target_term=term_data['target_term'],
                source_language=term_data['source_language'],
                target_language=term_data['target_language'],
                description=term_data['description'],
                term_id=i+1
            )
            corrector.term_corrections.append(term_correction)
        
        status_text.text("🚀 FORCE MODE ile işleniyor...")
        progress_bar.progress(30)
        
        # Setup basic logging for Streamlit
        import logging
        logger = logging.getLogger('streamlit_corrector')
        logger.setLevel(logging.INFO)
        
        # Process the file
        corrections_made, results = corrector.process_xliff_file(tmp_path, logger)
        
        progress_bar.progress(80)
        status_text.text("📊 Rapor oluşturuluyor...")
        
        # Read the corrected file
        with open(tmp_path, 'r', encoding='utf-8') as f:
            corrected_content = f.read()
        
        # Store results
        st.session_state.results = {
            'corrections_made': corrections_made,
            'detailed_results': results,
            'corrected_content': corrected_content,
            'stats': corrector.processing_stats,
            'original_filename': st.session_state.uploaded_file.name
        }
        
        progress_bar.progress(100)
        status_text.text("✅ İşlem tamamlandı!")
        
        st.session_state.processing_complete = True
        
        # Clean up temp file
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"❌ Dosya işlenirken hata: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ İşlem başarısız")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Universal Term Corrector</h1>
        <p>Evrensel format desteği ile AI destekli çok dilli terim düzeltme sistemi</p>
        <p><strong>SDL XLIFF • MemoQ XLIFF • Generic XLIFF • Force Mode AI</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Yapılandırma")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Claude API Anahtarı",
            type="password",
            help="Anthropic Claude API anahtarınızı girin"
        )
        
        if api_key:
            if st.session_state.corrector is None:
                with st.spinner("Corrector başlatılıyor..."):
                    create_corrector(api_key)
            
            if st.session_state.corrector:
                st.success("✅ Corrector başlatıldı")
        
        st.markdown("---")
        
        # Format support info
        st.markdown("""
        **🔧 Desteklenen Formatlar:**
        - SDL XLIFF (.sdlxliff)
        - MemoQ XLIFF (.mqxliff)
        - Standard XLIFF (.xliff, .xlf)
        - Generic XML çiftdil dosyaları
        
        **💪 Force Mode Özellikleri:**
        - AI gatekeeper yok
        - Uzman dilbilim kalitesi
        - Mükemmel gramer farkındalığı
        - Yapı korunması
        """)
    
    # Main content
    if not st.session_state.corrector:
        st.markdown("""
        <div class="warning-box">
            <h3>🔑 Kurulum Gerekli</h3>
            <p>Başlamak için lütfen yan çubuktan Claude API anahtarınızı girin.</p>
            <p>API anahtarını <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a> adresinden alabilirsiniz</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dosya Yükleme", "🔤 Terim Yönetimi", "🚀 İşleme", "📊 Sonuçlar"])
    
    with tab1:
        st.header("📁 Dosya Yükleme ve Format Tespiti")
        
        uploaded_file = st.file_uploader(
            "Çiftdil dosyası seçin",
            type=['sdlxliff', 'mqxliff', 'xliff', 'xlf', 'xml'],
            help="SDL XLIFF, MemoQ XLIFF veya standart XLIFF dosyaları yükleyin"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            
            st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
            
            # Detect format
            with st.spinner("🔍 Dosya formatı analiz ediliyor..."):
                format_info, tmp_path = detect_file_format(uploaded_file)
                
            if format_info:
                st.session_state.format_info = format_info
    
    with tab2:
        st.header("🔤 Terim Yönetimi")
        
        # Language selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Kaynak Dil",
                options=['en', 'de', 'fr', 'es', 'it', 'tr', 'ru', 'bg', 'ro', 'pl', 'cs'],
                index=0,
                key='source_lang'
            )
        
        with col2:
            target_lang = st.selectbox(
                "Hedef Dil",
                options=['en', 'de', 'fr', 'es', 'it', 'tr', 'ru', 'bg', 'ro', 'pl', 'cs'],
                index=5,  # Turkish as default target
                key='target_lang'
            )
        
        st.markdown("### ➕ Yeni Terim Düzeltmesi Ekle")
        
        # Term input form
        with st.form("add_term_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                source_term = st.text_input(
                    f"🔍 {source_lang.upper()} Terimi",
                    placeholder="Kaynak terimi girin..."
                )
            
            with col2:
                target_term = st.text_input(
                    f"✏️ {target_lang.upper()} Çevirisi",
                    placeholder="Hedef terimi girin..."
                )
            
            description = st.text_input(
                "📋 Açıklama (opsiyonel)",
                placeholder="Bu düzeltme için açıklama veya bağlam..."
            )
            
            submitted = st.form_submit_button("➕ Terim Ekle", use_container_width=True)
            
            if submitted and source_term and target_term:
                term = {
                    'source_term': source_term,
                    'target_term': target_term,
                    'description': description,
                    'source_language': source_lang,
                    'target_language': target_lang
                }
                st.session_state.terms.append(term)
                st.success(f"✅ Eklendi: '{source_term}' → '{target_term}'")
                st.rerun()
        
        # Display existing terms
        if st.session_state.terms:
            st.markdown("### 📋 Mevcut Terim Düzeltmeleri")
            
            for i, term in enumerate(st.session_state.terms):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
                    
                    with col1:
                        st.write(f"**{term['source_term']}**")
                    
                    with col2:
                        st.write(f"**{term['target_term']}**")
                    
                    with col3:
                        st.write(term['description'] or "Açıklama yok")
                    
                    with col4:
                        if st.button("🗑️", key=f"remove_{i}", help="Bu terimi kaldır"):
                            st.session_state.terms.pop(i)
                            st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("📝 Henüz terim düzeltmesi eklenmedi. Başlamak için yukarıdan terim ekleyin.")
    
    with tab3:
        st.header("🚀 Force Mode ile İşleme")
        
        # Pre-processing checks
        ready_checks = []
        ready_checks.append(("API Anahtarı", st.session_state.corrector is not None))
        ready_checks.append(("Dosya Yüklendi", st.session_state.uploaded_file is not None))
        ready_checks.append(("Terim Eklendi", len(st.session_state.terms) > 0))
        
        # Display readiness status
        col1, col2, col3 = st.columns(3)
        
        for i, (check_name, status) in enumerate(ready_checks):
            with [col1, col2, col3][i]:
                if status:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ {check_name}</h4>
                        <p>Hazır</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <h4>❌ {check_name}</h4>
                        <p>Gerekli</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        all_ready = all(status for _, status in ready_checks)
        
        if all_ready:
            st.markdown("""
            <div class="feature-box">
                <h3>💪 Force Mode Yapılandırması</h3>
                <p><strong>Mod:</strong> FORCE - Tüm örnekler düzeltilecek</p>
                <p><strong>Kalite:</strong> Uzman dilbilim analizi</p>
                <p><strong>Yapı:</strong> CAT aracı uyumluluğu için korunur</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Force Mode İşlemeyi Başlat", use_container_width=True, type="primary"):
                process_file()
        else:
            st.warning("⚠️ İşlemeden önce lütfen yukarıdaki tüm gereksinimleri tamamlayın.")
    
    with tab4:
        st.header("📊 İşleme Sonuçları")
        
        if st.session_state.processing_complete and st.session_state.results:
            results = st.session_state.results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['corrections_made']}</h3>
                    <p>Yapılan Düzeltme</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['stats']['total_units']}</h3>
                    <p>Toplam Birim</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                coverage_rate = (results['corrections_made'] / max(1, results['stats']['instances_found'])) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{coverage_rate:.1f}%</h3>
                    <p>Kapsam Oranı</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if results['detailed_results']:
                    avg_quality = sum(r.quality_score for r in results['detailed_results']) / len(results['detailed_results'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{avg_quality:.1%}</h3>
                        <p>Ort. Kalite</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download buttons
            st.markdown("### 📥 İndirmeler")
            col1, col2 = st.columns(2)
            
            with col1:
                corrected_content = results['corrected_content']
                original_filename = results['original_filename']
                
                name_parts = original_filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    corrected_filename = f"{name_parts[0]}_corrected.{name_parts[1]}"
                else:
                    corrected_filename = f"{original_filename}_corrected"
                
                st.download_button(
                    label="📥 Düzeltilmiş Dosyayı İndir",
                    data=corrected_content.encode('utf-8'),
                    file_name=corrected_filename,
                    mime="application/xml",
                    key="download_corrected"
                )
            
            with col2:
                report_data = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "original_file": results['original_filename'],
                        "processing_version": "Universal FORCE MODE Streamlit Edition",
                        "force_mode": True
                    },
                    "corrections_made": results['corrections_made'],
                    "statistics": results['stats'],
                    "term_corrections": st.session_state.terms
                }
                
                report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📊 İşleme Raporunu İndir",
                    data=report_json.encode('utf-8'),
                    file_name=f"correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_report"
                )
            
            # Detailed results
            if results['detailed_results']:
                st.markdown("### 📝 Detaylı Sonuçlar")
                
                results_data = []
                for result in results['detailed_results']:
                    results_data.append({
                        'Birim ID': result.unit_id,
                        'Uygulanan Düzeltmeler': ', '.join(result.applied_corrections),
                        'Kalite Skoru': f"{result.quality_score:.1%}",
                        'Güven': f"{result.confidence:.1%}"
                    })
                
                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
        
        else:
            st.info("📝 Henüz sonuç yok. Sonuçları görmek için İşleme sekmesinde bir dosya işleyin.")

if __name__ == "__main__":
    main()