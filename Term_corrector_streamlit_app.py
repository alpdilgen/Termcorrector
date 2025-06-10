import streamlit as st

# FIRST AND ONLY PAGE CONFIG
st.set_page_config(page_title="Universal Term Corrector", page_icon="🌍", layout="wide")

import tempfile
import os
import json
import pandas as pd
from datetime import datetime

def main():
    st.title("🌍 Universal Term Corrector")
    st.markdown("**AI destekli çok dilli terim düzeltme sistemi - FORCE MODE**")
    
    # Initialize session state
    if 'terms' not in st.session_state:
        st.session_state.terms = []
    if 'corrector' not in st.session_state:
        st.session_state.corrector = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    # Check if corrector module exists
    corrector_available = False
    import_status = ""
    
    try:
        from universal_term_corrector import UniversalTermCorrectorForce, TermCorrection, FileFormatInfo
        corrector_available = True
        import_status = "✅ Corrector modülü başarıyla yüklendi!"
    except ImportError as e:
        import_status = f"❌ Import hatası: {str(e)}"
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Yapılandırma")
        
        # Show status
        if corrector_available:
            st.success(import_status)
        else:
            st.error(import_status)
            with st.expander("🔍 Debug Bilgileri"):
                if os.path.exists("universal_term_corrector.py"):
                    st.info("📁 universal_term_corrector.py dosyası mevcut")
                    try:
                        with open("universal_term_corrector.py", "r", encoding="utf-8") as f:
                            content = f.read()
                        st.write(f"📏 Dosya boyutu: {len(content)} karakter")
                        
                        # Show first few lines
                        lines = content.split('\n')[:10]
                        st.write("**İlk 10 satır:**")
                        for i, line in enumerate(lines, 1):
                            st.code(f"{i:2d}: {line}")
                            
                    except Exception as e:
                        st.error(f"Dosya okunamadı: {e}")
                else:
                    st.error("📁 universal_term_corrector.py dosyası bulunamadı")
                    st.write("**Mevcut Python dosyaları:**")
                    for file in os.listdir("."):
                        if file.endswith('.py'):
                            st.write(f"- {file}")
        
        # API Key section
        st.subheader("🔑 API Anahtarı")
        api_key = st.text_input("Claude API Anahtarı", type="password")
        
        if api_key and corrector_available:
            try:
                if st.session_state.corrector is None:
                    st.session_state.corrector = UniversalTermCorrectorForce(api_key)
                st.success("✅ Corrector başlatıldı!")
            except Exception as e:
                st.error(f"❌ Corrector başlatılamadı: {e}")
        elif api_key:
            st.warning("API anahtarı girildi ama corrector modülü yok")
    
    # Main content
    if not corrector_available:
        st.error("❌ Universal Term Corrector modülü yüklenemedi.")
        st.info("🔧 Lütfen universal_term_corrector.py dosyasının doğru yüklendiğinden emin olun.")
        return
    
    if not st.session_state.corrector:
        st.warning("⚠️ Devam etmek için lütfen Claude API anahtarınızı yan çubuktan girin.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dosya Yükleme", "🔤 Çoklu Terim Yönetimi", "🚀 İşleme", "📊 Sonuçlar"])
    
    with tab1:
        st.header("📁 Dosya Yükleme ve Format Tespiti")
        
        uploaded_file = st.file_uploader(
            "XLIFF dosyası seçin",
            type=['sdlxliff', 'mqxliff', 'xliff', 'xlf', 'xml'],
            help="SDL XLIFF, MemoQ XLIFF veya standart XLIFF dosyaları yükleyin"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
            
            # Save to temp file and detect format
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                corrector = st.session_state.corrector
                format_info = corrector.detect_bilingual_format(tmp_path)
                
                # Display format information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📋 Format", f"{format_info.format_type.upper()}")
                    st.caption(f"Versiyon: {format_info.version}")
                
                with col2:
                    st.metric("🔧 Yapı", format_info.structure_type.replace('_', ' ').title())
                
                with col3:
                    st.metric("✨ Özellikler", len(format_info.special_features))
                
                if format_info.special_features:
                    st.write("**Özel Özellikler:**")
                    for feature in format_info.special_features:
                        st.write(f"• {feature.replace('_', ' ').title()}")
                
                # Try to detect languages
                source_lang, target_lang = corrector.detect_languages_from_universal_format(tmp_path)
                if source_lang and target_lang:
                    st.info(f"🌐 Tespit edilen diller: {source_lang.upper()} → {target_lang.upper()}")
                
            except Exception as e:
                st.error(f"❌ Format tespitinde hata: {e}")
            finally:
                os.unlink(tmp_path)
    
    with tab2:
        st.header("🔤 Çoklu Terim Yönetimi")
        st.markdown("**FORCE MODE**: Tüm terimler otomatik olarak düzeltilecek!")
        
        # Language selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Kaynak Dil",
                options=['en', 'de', 'fr', 'es', 'it', 'tr', 'ru', 'bg', 'ro', 'pl', 'cs', 'sk', 'hr', 'sl', 'hu'],
                index=0,
                key='source_lang'
            )
        
        with col2:
            target_lang = st.selectbox(
                "Hedef Dil",
                options=['en', 'de', 'fr', 'es', 'it', 'tr', 'ru', 'bg', 'ro', 'pl', 'cs', 'sk', 'hr', 'sl', 'hu'],
                index=5,  # Turkish as default
                key='target_lang'
            )
        
        # Add new term form
        st.subheader("➕ Yeni Terim Ekle")
        with st.form("add_term_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                source_term = st.text_input(
                    f"🔍 {source_lang.upper()} Terimi",
                    placeholder="Değiştirilecek terim..."
                )
            
            with col2:
                target_term = st.text_input(
                    f"✏️ {target_lang.upper()} Karşılığı",
                    placeholder="Yeni terim..."
                )
            
            description = st.text_input(
                "📋 Açıklama (opsiyonel)",
                placeholder="Bu düzeltme hakkında notlar..."
            )
            
            submitted = st.form_submit_button("➕ Terim Ekle", use_container_width=True)
            
            if submitted:
                if source_term and target_term:
                    # Create TermCorrection object
                    new_term = {
                        'source_term': source_term,
                        'target_term': target_term,
                        'source_language': source_lang,
                        'target_language': target_lang,
                        'description': description,
                        'term_id': len(st.session_state.terms) + 1
                    }
                    
                    st.session_state.terms.append(new_term)
                    st.success(f"✅ Eklendi: '{source_term}' → '{target_term}'")
                    st.rerun()
                else:
                    st.error("❌ Hem kaynak hem hedef terim gerekli!")
        
        # Display existing terms
        if st.session_state.terms:
            st.subheader(f"📋 Kayıtlı Terimler ({len(st.session_state.terms)} adet)")
            
            # Create a dataframe for better display
            terms_data = []
            for i, term in enumerate(st.session_state.terms):
                terms_data.append({
                    'ID': term['term_id'],
                    'Kaynak': term['source_term'],
                    'Hedef': term['target_term'],
                    'Dil Çifti': f"{term['source_language']} → {term['target_language']}",
                    'Açıklama': term['description'] or "Yok"
                })
            
            df = pd.DataFrame(terms_data)
            st.dataframe(df, use_container_width=True)
            
            # Bulk operations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Tümünü Temizle", type="secondary"):
                    st.session_state.terms = []
                    st.rerun()
            
            with col2:
                # Export terms as JSON
                if st.button("📤 Export JSON"):
                    terms_json = json.dumps(st.session_state.terms, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 İndir",
                        data=terms_json,
                        file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                # Import terms
                uploaded_terms = st.file_uploader("📥 JSON Import", type=['json'], key="import_terms")
                if uploaded_terms:
                    try:
                        imported_terms = json.loads(uploaded_terms.getvalue().decode('utf-8'))
                        st.session_state.terms.extend(imported_terms)
                        st.success(f"✅ {len(imported_terms)} terim import edildi!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Import hatası: {e}")
            
            # Individual term removal
            st.subheader("🔧 Terim Düzenleme")
            for i, term in enumerate(st.session_state.terms):
                with st.expander(f"Terim {term['term_id']}: {term['source_term']} → {term['target_term']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Kaynak:** {term['source_term']}")
                        st.write(f"**Hedef:** {term['target_term']}")
                        st.write(f"**Diller:** {term['source_language']} → {term['target_language']}")
                        if term['description']:
                            st.write(f"**Açıklama:** {term['description']}")
                    
                    with col2:
                        if st.button(f"🗑️ Sil", key=f"delete_{i}"):
                            st.session_state.terms.pop(i)
                            st.rerun()
        
        else:
            st.info("📝 Henüz terim eklenmedi. Yukarıdaki formu kullanarak terim ekleyin.")
    
    with tab3:
        st.header("🚀 FORCE MODE İşleme")
        
        # Prerequisites check
        prerequisites = [
            ("API Anahtarı", st.session_state.corrector is not None),
            ("Dosya Yüklendi", st.session_state.uploaded_file is not None),
            ("Terim Var", len(st.session_state.terms) > 0)
        ]
        
        # Display status
        col1, col2, col3 = st.columns(3)
        for i, (name, status) in enumerate(prerequisites):
            with [col1, col2, col3][i]:
                if status:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
        
        all_ready = all(status for _, status in prerequisites)
        
        if all_ready:
            st.success("🎉 Tüm gereksinimler karşılandı!")
            
            # Show processing summary
            st.subheader("📋 İşleme Özeti")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📁 Dosya", st.session_state.uploaded_file.name)
                st.metric("🔤 Terim Sayısı", len(st.session_state.terms))
            
            with col2:
                st.metric("🌐 Dil Çiftleri", len(set(f"{t['source_language']}-{t['target_language']}" for t in st.session_state.terms)))
                st.metric("💪 Mod", "FORCE")
            
            # Processing button
            if st.button("🚀 FORCE MODE İşlemeyi Başlat", type="primary", use_container_width=True):
                process_file_with_terms()
        
        else:
            st.warning("⚠️ Lütfen tüm gereksinimleri karşılayın:")
            for name, status in prerequisites:
                if not status:
                    st.write(f"• {name} eksik")
    
    with tab4:
        st.header("📊 İşleme Sonuçları")
        
        if st.session_state.processing_results:
            results = st.session_state.processing_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💪 Düzeltme", results['corrections_made'])
            
            with col2:
                st.metric("📊 Toplam Birim", results.get('total_units', 0))
            
            with col3:
                instances = results.get('instances_found', 0)
                coverage = (results['corrections_made'] / max(1, instances)) * 100
                st.metric("📈 Kapsam", f"{coverage:.1f}%")
            
            with col4:
                if results.get('detailed_results'):
                    avg_quality = sum(r.quality_score for r in results['detailed_results']) / len(results['detailed_results'])
                    st.metric("🎯 Kalite", f"{avg_quality:.1%}")
            
            # Download section
            st.subheader("📥 İndirmeler")
            col1, col2 = st.columns(2)
            
            with col1:
                if results.get('corrected_content'):
                    original_name = st.session_state.uploaded_file.name
                    name_parts = original_name.rsplit('.', 1)
                    corrected_name = f"{name_parts[0]}_corrected.{name_parts[1]}" if len(name_parts) == 2 else f"{original_name}_corrected"
                    
                    st.download_button(
                        label="📥 Düzeltilmiş Dosyayı İndir",
                        data=results['corrected_content'].encode('utf-8'),
                        file_name=corrected_name,
                        mime="application/xml"
                    )
            
            with col2:
                # Generate report
                report_data = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "file": st.session_state.uploaded_file.name,
                        "force_mode": True
                    },
                    "corrections": results['corrections_made'],
                    "terms_used": st.session_state.terms,
                    "statistics": results.get('stats', {})
                }
                
                report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📊 İşleme Raporu İndir",
                    data=report_json,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Detailed results
            if results.get('detailed_results'):
                st.subheader("📝 Detaylı Sonuçlar")
                
                detailed_data = []
                for result in results['detailed_results']:
                    detailed_data.append({
                        'Birim ID': result.unit_id,
                        'Düzeltmeler': ', '.join(result.applied_corrections),
                        'Kalite': f"{result.quality_score:.1%}",
                        'Güven': f"{result.confidence:.1%}"
                    })
                
                if detailed_data:
                    df_detailed = pd.DataFrame(detailed_data)
                    st.dataframe(df_detailed, use_container_width=True)
            
            # Statistics
            if results.get('stats'):
                with st.expander("📈 İstatistikler"):
                    st.json(results['stats'])
        
        else:
            st.info("📝 Henüz işleme yapılmadı. İşleme sekmesinde dosyayı işleyin.")

def process_file_with_terms():
    """Process the uploaded file with all terms"""
    
    # Show progress
    progress_container = st.container()
    with progress_container:
        st.info("🔄 İşleme başlatılıyor...")
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
        
        # Clear previous term corrections and add current ones
        corrector.term_corrections = []
        
        # Import TermCorrection class
        from universal_term_corrector import TermCorrection
        
        # Convert session terms to TermCorrection objects
        for term_data in st.session_state.terms:
            term_correction = TermCorrection(
                source_term=term_data['source_term'],
                target_term=term_data['target_term'],
                source_language=term_data['source_language'],
                target_language=term_data['target_language'],
                description=term_data['description'],
                term_id=term_data['term_id']
            )
            corrector.term_corrections.append(term_correction)
        
        status_text.text(f"🚀 FORCE MODE: {len(corrector.term_corrections)} terim ile işleniyor...")
        progress_bar.progress(30)
        
        # Setup logging
        import logging
        logger = logging.getLogger('streamlit_corrector')
        logger.setLevel(logging.INFO)
        
        # Process the file
        corrections_made, detailed_results = corrector.process_xliff_file(tmp_path, logger)
        
        progress_bar.progress(80)
        status_text.text("📊 Sonuçlar hazırlanıyor...")
        
        # Read corrected file
        with open(tmp_path, 'r', encoding='utf-8') as f:
            corrected_content = f.read()
        
        # Store results
        st.session_state.processing_results = {
            'corrections_made': corrections_made,
            'detailed_results': detailed_results,
            'corrected_content': corrected_content,
            'stats': corrector.processing_stats,
            'total_units': corrector.processing_stats.get('total_units', 0),
            'instances_found': corrector.processing_stats.get('instances_found', 0)
        }
        
        progress_bar.progress(100)
        status_text.text("✅ İşlem tamamlandı!")
        
        # Show success message
        st.success(f"🎉 İşlem başarıyla tamamlandı! {corrections_made} düzeltme yapıldı.")
        
        # Clean up
        os.unlink(tmp_path)
        
        # Switch to results tab
        st.info("📊 Sonuçları görüntülemek için 'Sonuçlar' sekmesine geçin.")
        
    except Exception as e:
        st.error(f"❌ İşleme sırasında hata: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ İşlem başarısız")

if __name__ == "__main__":
    main()