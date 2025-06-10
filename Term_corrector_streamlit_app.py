import streamlit as st

# FIRST AND ONLY PAGE CONFIG
st.set_page_config(page_title="Universal Term Corrector", page_icon="🌍")

import tempfile
import os
import json
from datetime import datetime

def main():
    st.title("🌍 Universal Term Corrector")
    st.write("AI destekli çok dilli terim düzeltme sistemi")
    
    # Check if corrector module exists
    corrector_available = False
    import_status = ""
    
    try:
        from universal_term_corrector import UniversalTermCorrectorForce
        corrector_available = True
        import_status = "✅ Corrector modülü başarıyla yüklendi!"
    except ImportError as e:
        import_status = f"❌ Import hatası: {str(e)}"
    
    # Show status
    if corrector_available:
        st.success(import_status)
    else:
        st.error(import_status)
        
        # Debug info
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
        st.success("API anahtarı girildi!")
    elif api_key:
        st.warning("API anahtarı girildi ama corrector modülü yok")
    
    # File upload section
    st.subheader("📁 Dosya Yükleme")
    uploaded_file = st.file_uploader(
        "XLIFF dosyası seçin",
        type=['sdlxliff', 'mqxliff', 'xliff', 'xlf', 'xml']
    )
    
    if uploaded_file:
        st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
    
    # Basic term input
    st.subheader("🔤 Terim Düzeltmesi")
    col1, col2 = st.columns(2)
    
    with col1:
        source_term = st.text_input("Kaynak terim")
    
    with col2:
        target_term = st.text_input("Hedef terim")
    
    if source_term and target_term:
        st.info(f"Düzeltme: '{source_term}' → '{target_term}'")
    
    # Process button
    if st.button("🚀 İşle", type="primary"):
        if not api_key:
            st.error("❌ API anahtarı gerekli")
        elif not corrector_available:
            st.error("❌ Corrector modülü yok")
        elif not uploaded_file:
            st.error("❌ Dosya yüklenmedi")
        elif not source_term or not target_term:
            st.error("❌ Terimler girilmedi")
        else:
            st.success("🎉 Tüm gereksinimler karşılandı! (İşleme henüz implement edilmedi)")
    
    # Status footer
    st.markdown("---")
    st.caption("Universal Term Corrector - Streamlit Edition")

if __name__ == "__main__":
    main()