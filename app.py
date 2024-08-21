import streamlit as st

# Set page configuration
st.set_page_config(page_title="Hoax Detection Dashboard", layout="wide")
st.title("Dashboard Deteksi Berita Hoax")

from home import show_home
from deteksi_content import show_deteksi_konten
from deteksi_upload import show_deteksi_upload

# Create tabs
tab1, tab2, tab3 = st.tabs(["Home", "Deteksi Konten", "Deteksi File"])

with tab1:
    show_home()

with tab2:
    show_deteksi_konten()

with tab3:
    show_deteksi_upload()