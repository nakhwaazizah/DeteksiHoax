import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from test import predict_hoax, evaluate_model_performance
from load_model import load_model
from styles import COMMON_CSS

def load_data(file):
    return pd.read_csv(file)

def show_deteksi_upload():
    st.markdown(COMMON_CSS, unsafe_allow_html=True)
    
    st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Pilih Model</h6>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "",
        [
            "cahya/bert-base-indonesian-522M",
            "indobenchmark/indobert-base-p2",
            "indolem/indobert-base-uncased",
            "mdhugol/indonesia-bert-sentiment-classification"
        ],
        key="model_selector_upload"
    )

    # Load the selected model
    tokenizer, model = load_model(selected_model)

    # File uploader
    st.markdown("<h6 style='font-size: 14px; margin-bottom: -200px;'>Unggah File Disini</h6>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="csv")

    # Initialize session state for corrections
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Periksa apakah file diunggah
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df.index = df.index + 1

        st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Data yang Diunggah</h6>", unsafe_allow_html=True)

        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gridOptions = grid_options.build()

        AgGrid(
            df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            use_container_width=True
        )

        if st.button("Deteksi", key="detect_upload"):
            try:
                # Run the prediction and update the dataframe
                df['Result'] = df.apply(lambda row: predict_hoax(row['Title'], row['Content']), axis=1)
                df['Correction'] = False 

                # Save detection results in session state
                st.session_state.df = df.copy()
                st.success("Deteksi berhasil!")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {e}")

    # Display detection results if available
    if st.session_state.df is not None:

        accuracy, precision, recall, f1 = evaluate_model_performance(st.session_state.df, tokenizer, model)
        performance_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)]
        })
        st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Performansi Model</h6>", unsafe_allow_html=True)
        st.dataframe(performance_df, use_container_width=True, hide_index=True)

        st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Hasil Deteksi</h6>", unsafe_allow_html=True)

        # Reorder columns to bring 'Correction' and 'Result' to the front
        cols = ['Correction', 'Result'] + [col for col in st.session_state.df.columns if col not in ['Correction', 'Result']]
        df_reordered = st.session_state.df[cols]

        # Simplify the AgGrid setup
        grid_options = GridOptionsBuilder.from_dataframe(df_reordered)
        grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        grid_options.configure_default_column(editable=True, groupable=True)
        grid_options.configure_column('Content', width=50)
        gridOptions = grid_options.build()

        # Use AgGrid to display the DataFrame
        grid_response = AgGrid(
            st.session_state.df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.VALUE_CHANGED
        )

        # Update session state with the edited DataFrame
        if grid_response['data'] is not None:
            edited_df = pd.DataFrame(grid_response['data'])
            st.session_state.df = edited_df.copy()
            corrected_df = edited_df[edited_df['Correction']].copy()

            # Handle corrections
            edited_df['Final_Result'] = edited_df.apply(lambda row: 
                'HOAX' if (row['Result'] == 'NON-HOAX' and row['Correction']) else 
                ('NON-HOAX' if (row['Result'] == 'HOAX' and row['Correction']) else row['Result']), 
                axis=1
            )

            st.session_state.df = edited_df.copy()

            if not corrected_df.empty:
                # Add Final_Result column only to the corrected DataFrame
                corrected_df['Final_Result'] = corrected_df.apply(lambda row: 
                    'HOAX' if (row['Result'] == 'NON-HOAX' and row['Correction']) else 
                    ('NON-HOAX' if (row['Result'] == 'HOAX' and row['Correction']) else row['Result']), 
                    axis=1
                )

                st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Data yang Dikoreksi</h6>", unsafe_allow_html=True)
                st.dataframe(corrected_df.drop(columns=['Correction']), use_container_width=True)
            else:
                st.write("Tidak ada data yang dikoreksi.")
        
        # Save corrected data
        if st.button("Simpan", key="corrected_data"):
            if 'df' in st.session_state:
                corrected_df = st.session_state.df[st.session_state.df['Correction']].drop(columns=['Correction'])
                if not corrected_df.empty:
                    corrected_df.to_csv("corrected_df.csv", index=False)
                    st.success("Data telah disimpan.")
                    st.session_state.corrected_df = corrected_df
                else:
                    st.warning("Tidak ada data yang dikoreksi untuk disimpan.")
            else:
                st.warning("Data deteksi tidak ditemukan.")
