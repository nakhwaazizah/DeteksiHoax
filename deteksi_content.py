import streamlit as st
from datetime import datetime
import pandas as pd
from lime.lime_text import LimeTextExplainer
from test import predict_hoax, predict_proba_for_lime
import streamlit.components.v1 as components
from load_model import load_model
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from styles import COMMON_CSS

def show_deteksi_konten():
    st.markdown(COMMON_CSS, unsafe_allow_html=True)

    if 'correction' not in st.session_state:
        st.session_state.correction = None
    if 'detection_result' not in st.session_state:
        st.session_state.detection_result = None
    if 'lime_explanation' not in st.session_state:
        st.session_state.lime_explanation = None
    if 'headline' not in st.session_state:
        st.session_state.headline = ""
    if 'content' not in st.session_state:
        st.session_state.content = ""
    if 'is_correct' not in st.session_state:
        st.session_state.is_correct = None

    # Dropdown for selecting a model
    st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Pilih Model</h6>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "",
        [
            "cahya/bert-base-indonesian-522M",
            "indobenchmark/indobert-base-p2",
            "indolem/indobert-base-uncased",
            "mdhugol/indonesia-bert-sentiment-classification"
        ],
        key="model_selector_content"
    )

    # Load the selected model
    tokenizer, model = load_model(selected_model)

    st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Masukkan Judul Berita :</h6>", unsafe_allow_html=True)
    st.session_state.headline = st.text_input("", value=st.session_state.headline)

    st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Masukkan Konten Berita :</h6>", unsafe_allow_html=True)
    st.session_state.content = st.text_area("", value=st.session_state.content)

    # Detection button
    if st.button("Deteksi", key="detect_content"):
        st.session_state.detection_result = predict_hoax(st.session_state.headline, st.session_state.content)
        st.success(f"Prediksi: {st.session_state.detection_result}")

        # Prepare the text for LIME
        lime_texts = [f"{st.session_state.headline} [SEP] {st.session_state.content}"]

        # Add a spinner and progress bar to indicate processing
        with st.spinner("Sedang memproses LIME, harap tunggu..."):
            # Explain the prediction
            explainer = LimeTextExplainer(class_names=['NON-HOAX', 'HOAX'])
            explanation = explainer.explain_instance(lime_texts[0], predict_proba_for_lime, num_features=5, num_samples=1000)

            # Save the LIME explanation in session state
            st.session_state.lime_explanation = explanation.as_html()

    # Display the detection result and LIME explanation if available
    # if st.session_state.detection_result is not None:
    #     st.success(f"Prediksi: {st.session_state.detection_result}")
    if st.session_state.lime_explanation:
        lime_html = st.session_state.lime_explanation

        # Inject CSS for font size adjustment
        lime_html = f"""
        <style>
        .lime-text-explanation, .lime-highlight, .lime-classification, 
        .lime-text-explanation * {{
            font-size: 14px !important;
        }}
        </style>
        <div class="lime-text-explanation">
            {lime_html}
        </div>
        """
        components.html(lime_html, height=200, scrolling=True)

    # Display a radio button asking if the detection result is correct
    if st.session_state.detection_result is not None:
        st.markdown("<h6 style='font-size: 16px; margin-bottom: -150px;'>Apakah hasil deteksi sudah benar?</h6>", unsafe_allow_html=True)
        st.session_state.is_correct = st.radio("", ("Ya", "Tidak"))

        if st.session_state.is_correct == "Ya":
            st.success("Deteksi sudah benar.")
        else:
            # Determine the correction based on the prediction
            st.session_state.correction = "HOAX" if st.session_state.detection_result == "NON-HOAX" else "NON-HOAX"

            # Display the correction DataFrame
            correction_data = pd.DataFrame([{
                'Title': st.session_state.headline,
                'Content': st.session_state.content,
                'Prediction': st.session_state.detection_result,
                'Correction': st.session_state.correction
            }])

            # Save button
            if st.button("Simpan"):
                # Display the correction as text
                st.markdown(f"<p style='font-size: 14px;'>Title         : {st.session_state.headline}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14px;'>Content       : {st.session_state.content}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14px;'>Prediction    : {st.session_state.detection_result}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14px;'>Correction    : {st.session_state.correction}</p>", unsafe_allow_html=True)
                st.success("Koreksi telah disimpan.")
