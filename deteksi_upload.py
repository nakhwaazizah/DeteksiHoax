import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from test import predict_hoax, evaluate_model_performance
from load_model import load_model
from styles import COMMON_CSS
from google.cloud import storage
from io import StringIO
import os
from datetime import datetime
import pytz

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Lenovo\Downloads\DasboardBert\inbound-source-431806-g7-e49e388ce0be.json"

destination_file_name = '/tmp/inbound-source-431806-g7-e49e388ce0be.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = destination_file_name
bucket_name = 'dashboardhoax-bucket'
source_blob_name = 'dashboardhoax-bucket/inbound-source-431806-g7-e49e388ce0be.json'

def download_json_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

# Unduh file JSON dari GCS
download_json_from_gcs(bucket_name, source_blob_name, destination_file_name)

def save_corrections_to_gcs(bucket_name, file_name, correction_data):
    client = storage.Client()
    bucket = client.bucket("dashboardhoax-bucket")
    blob = bucket.blob("koreksi_pengguna_file.csv")

    if blob.exists():
        existing_data = blob.download_as_string().decode('utf-8')
        existing_df = pd.read_csv(StringIO(existing_data))
    else:
        existing_df = pd.DataFrame(columns=['Timestamp', 'Label_id', 'Label', 'Title', 'Content', 'Fact', 'References', 'Classification', 'Datasource', 'Result_Detection', 'Result_Correction'])

    new_data_df = pd.DataFrame(correction_data)
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    updated_csv_data = updated_df.to_csv(index=False)
    blob.upload_from_string(updated_csv_data, content_type='text/csv')

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

    tokenizer, model = load_model(selected_model)

    st.markdown("<h6 style='font-size: 14px; margin-bottom: -200px;'>Unggah File Disini</h6>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="csv")

    if 'df' not in st.session_state:
        st.session_state.df = None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df.index = df.index + 1

        st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Data yang Diunggah</h6>", unsafe_allow_html=True)

        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        grid_options.configure_default_column(cellStyle={'fontSize': '12px'})
        gridOptions = grid_options.build()

        AgGrid(
            df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            use_container_width=True
        )

        if st.button("Deteksi", key="detect_upload"):
            try:
                df['Result_Detection'] = df.apply(lambda row: predict_hoax(row['Title'], row['Content']), axis=1)
                df['Correction'] = False 
                st.session_state.df = df.copy()
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi: {e}")

    if st.session_state.df is not None:
        if 'Label' in st.session_state.df.columns:
            accuracy, precision, recall, f1 = evaluate_model_performance(st.session_state.df, tokenizer, model)
            performance_text = (
                f"*Performansi Model*\n\n"
                f"*Accuracy:* {round(accuracy, 2)}&nbsp;&nbsp;"
                f"*Precision:* {round(precision, 2)}&nbsp;&nbsp;"
                f"*Recall:* {round(recall, 2)}&nbsp;&nbsp;"
                f"*F1 Score:* {round(f1, 2)}"
            )
            st.success(performance_text)

        st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Hasil Deteksi</h6>", unsafe_allow_html=True)

        cols = ['Correction', 'Result_Detection'] + [col for col in st.session_state.df.columns if col not in ['Correction', 'Result_Detection', 'Label_id']]
        df_reordered = st.session_state.df[cols]

        grid_options = GridOptionsBuilder.from_dataframe(df_reordered)
        grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        grid_options.configure_default_column(editable=True, groupable=True)
        grid_options.configure_default_column(cellStyle={'fontSize': '12px'})
        gridOptions = grid_options.build()

        grid_response = AgGrid(
            st.session_state.df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.VALUE_CHANGED
        )

        if grid_response['data'] is not None:
            edited_df = pd.DataFrame(grid_response['data'])
            st.session_state.df = edited_df.copy()
            corrected_df = edited_df[edited_df['Correction']].copy()

            edited_df['Result_Correction'] = edited_df.apply(lambda row: 
                'HOAX' if (row['Result_Detection'] == 'NON-HOAX' and row['Correction']) else 
                ('NON-HOAX' if (row['Result_Detection'] == 'HOAX' and row['Correction']) else row['Result_Detection']), 
                axis=1
            )

            st.session_state.df = edited_df.copy()

            if not corrected_df.empty:
                expected_cols = ['Timestamp', 'Result_Detection', 'Result_Correction', 'Label_id', 'Label', 'Title', 'Content', 'Fact', 'References', 'Classification', 'Datasource']
                existing_cols = [col for col in expected_cols if col in corrected_df.columns]

                # Tambahkan Timestamp hanya untuk penyimpanan
                wib = pytz.timezone('Asia/Jakarta')
                corrected_df['Timestamp'] = datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S')

                corrected_df_to_display = corrected_df[existing_cols]

                st.markdown("<h6 style='font-size: 16px; margin-bottom: 0;'>Data yang Dikoreksi</h6>", unsafe_allow_html=True)
                st.dataframe(corrected_df_to_display, use_container_width=True, hide_index=True)
            else:
                st.write("Tidak ada data yang dikoreksi.")
        
        if st.button("Simpan", key="corrected_data"):
            if 'df' in st.session_state:
                corrected_df = st.session_state.df[st.session_state.df['Correction']].copy()
                wib = pytz.timezone('Asia/Jakarta')
                corrected_df['Timestamp'] = datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S')
                corrected_df = corrected_df.drop(columns=['Correction'])

                if not corrected_df.empty:
                    bucket_name = "your-bucket-name"
                    file_name = "corrected_upload_data.csv"
                    
                    correction_data = corrected_df.to_dict(orient='records')
                    
                    save_corrections_to_gcs(bucket_name, file_name, correction_data)
                    
                    st.success("Data telah disimpan.")
                    st.session_state.corrected_df = corrected_df
                else:
                    st.warning("Tidak ada data yang dikoreksi untuk disimpan.")
            else:
                st.warning("Data deteksi tidak ditemukan.")
