import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from test import predict_hoax, predict_proba_for_lime
import streamlit.components.v1 as components
from load_model import load_model

st.title("Dashboard Deteksi Berita Hoax")

tab1, tab2 = st.tabs(["Home", "Deteksi Hoax"])

with tab1:
    # Load the dataset
    df = pd.read_csv("data.csv")

    # Convert 'Tanggal' to datetime
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')

    # Number of records per datasource
    datasource_label_counts = df.groupby(['Datasource', 'Label']).size().reset_index(name='counts')
    fig_datasource = px.bar(datasource_label_counts, x='Datasource', y='counts', color='Label', barmode='group', 
                            title="Jumlah Hoax & Non-Hoax per Datasource")
    fig_datasource.update_layout(height=280, width=280)

    # Number of records per label
    label_counts = df["Label"].value_counts()
    fig_label_pie = px.pie(values=label_counts.values, names=label_counts.index, title="Perbandingan Jumlah Label")
    fig_label_pie.update_layout(height=300, width=300)

    # Hoax and Non-Hoax per year
    df['Year'] = df['Tanggal'].dt.year
    yearly_counts = df.groupby(['Year', 'Label']).size().unstack(fill_value=0)
    fig_yearly = px.line(yearly_counts, x=yearly_counts.index, y=['HOAX', 'NON-HOAX'], title='Hoax & Non-Hoax per Tahun')
    fig_yearly.update_layout(height=250, width=250)
    fig_yearly.update_traces(line=dict(color='red'), selector=dict(name='HOAX'))
    fig_yearly.update_traces(line=dict(color='green'), selector=dict(name='NON-HOAX'))

    # Word Cloud
    text = " ".join(content for content in df.Content)
    wordcloud = WordCloud(width=280, height=120, background_color='white').generate(text)
        
    fig_wordcloud, ax = plt.subplots(figsize=(3, 1.5), dpi=100)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud of Content", fontsize=8, fontweight='bold')
    plt.tight_layout(pad=0)

    # Add custom CSS to reduce the margin between columns
    st.markdown(
        """
        <style>
        div[data-testid="column"] {
            margin-left: -1em;
            margin-right: -1em; 
            padding: 0 0.5em;
        }
        .css-18e3th9 {
            padding-top: 0.5em;
            padding-bottom: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Arrange visualizations into two per row
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.plotly_chart(fig_datasource, use_container_width=True)
        st.plotly_chart(fig_label_pie, use_container_width=True)

    with col2:
        st.plotly_chart(fig_yearly, use_container_width=True)
        st.pyplot(fig_wordcloud)

with tab2:
    # Dropdown for selecting a model
    selected_model = st.selectbox(
        "Pilih Model",
        [
            "cahya/bert-base-indonesian-522M",
            "indobenchmark/indobert-base-p2",
            "indolem/indobert-base-uncased",
            "mdhugol/indonesia-bert-sentiment-classification"
        ]
    )

    # Load the selected model
    tokenizer, model = load_model(selected_model)

    # Text input for the headline
    headline = st.text_input("Masukkan judul berita")

    # Text area for the article content
    content = st.text_area("Masukkan konten berita")

    # Detection button
    if st.button("Deteksi"):
        # Get prediction label
        label = predict_hoax(headline, content)
        st.write(f"Prediksi: {label}")
