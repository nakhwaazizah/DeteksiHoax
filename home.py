import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Caching data loading
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df_evaluasi = pd.read_csv("Evaluasi Model.csv")
    df_mafindo = pd.read_csv("mafindo_mix_llm.csv")
    return df, df_evaluasi, df_mafindo

# Caching WordCloud generation
@st.cache_resource
def generate_wordcloud(text, colormap, stopwords):
    wordcloud = WordCloud(width=500, height=200, background_color='white', colormap=colormap, stopwords=stopwords).generate(text)
    return wordcloud

def show_home():
    # Load the dataset
    df, df_evaluasi, df_mafindo = load_data()

    # Convert 'Tanggal' to datetime
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
    df['Year'] = df['Tanggal'].dt.year

    # Convert text columns to string to avoid type errors
    df['Content'] = df['Content'].astype(str)

    # Define additional stopwords
    additional_stopwords = {"dan", "di", "yang", "ke", "dari", "untuk", "pada", "adalah", "sebuah", "dengan", "tersebut", "ini", "itu", "atau", "dalam", "juga", "adalah"}

    # Combine default stopwords with additional stopwords
    combined_stopwords = set(STOPWORDS).union(additional_stopwords)


    # Row with 4 visualizations
    col1, col2, col3, col4 = st.columns([1.5, 2.5, 1.5, 2.5])

    # Visualization 1: Bar chart for Hoax vs Non-Hoax using Plotly
    with col1:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Hoax vs Non-Hoax</h6>", unsafe_allow_html=True)
        df_label_counts = df['Label'].value_counts().reset_index()
        df_label_counts.columns = ['Label', 'Jumlah']
        bar_chart_label = px.bar(df_label_counts, x='Label', y='Jumlah', color='Label',
                                color_discrete_map={'HOAX': 'red', 'NON-HOAX': 'green'})
        bar_chart_label.update_layout(
            width=200, height=150, xaxis_title='Label', yaxis_title='Jumlah',
            xaxis_title_font_size=10, yaxis_title_font_size=10,
            xaxis_tickfont_size=8, yaxis_tickfont_size=8, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(bar_chart_label, use_container_width=False)

    # Visualization 2: Bar chart for Hoax vs Non-Hoax per Data Source using Plotly
    with col2:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Hoax vs Non-Hoax per Data Source</h6>", unsafe_allow_html=True)
        datasource_label_counts = df.groupby(['Datasource', 'Label']).size().reset_index(name='counts')
        fig_datasource = px.bar(datasource_label_counts, x='Datasource', y='counts', color='Label', barmode='group',
                               color_discrete_map={'HOAX': 'red', 'NON-HOAX': 'green'})
        fig_datasource.update_layout(
            width=500, height=150, xaxis_title='Datasource', yaxis_title='Jumlah',
            xaxis_title_font_size=10, yaxis_title_font_size=10,
            xaxis_tickfont_size=8, yaxis_tickfont_size=8, xaxis_tickangle=0,
            margin=dict(t=10, b=10, l=10, r=50),
            legend=dict(
                font=dict(size=8),  # Smaller font size for the legend
                traceorder='normal',
                orientation='v',  # Vertical orientation of the legend
                title_text='Label',  # Title for the legend
                yanchor='top', y=1, xanchor='left', x=1.05,  # Adjust position of the legend
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
                bordercolor='rgba(0, 0, 0, 0)'  # No border color
            ),
            showlegend=True
        )
        st.plotly_chart(fig_datasource, use_container_width=False)
    
    # Visualization 3: Line chart for Hoax per Year using Plotly
    with col3:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Hoax per Tahun</h6>", unsafe_allow_html=True)
    
    # Filter data to include only years up to 2023
        hoax_per_year = df[(df['Label'] == 'HOAX') & (df['Year'] <= 2023)].groupby('Year').size().reset_index(name='count')
    
        line_chart_hoax = px.line(hoax_per_year, x='Year', y='count', line_shape='linear',
                              color_discrete_sequence=['red'])
        line_chart_hoax.update_layout(
            width=200, height=150, xaxis_title='Tahun', yaxis_title='Jumlah Hoax',
            xaxis_title_font_size=10, yaxis_title_font_size=10,
            xaxis_tickfont_size=8, yaxis_tickfont_size=8, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(line_chart_hoax, use_container_width=False)

    
    # Visualization 4: Bar chart for Topics per Year using Plotly
    with col4:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Topics per Tahun</h6>", unsafe_allow_html=True)
        df_mafindo['Tanggal'] = pd.to_datetime(df_mafindo['Tanggal'], format='%d/%m/%Y')
        df_mafindo['Year'] = df_mafindo['Tanggal'].dt.year

        # Filter the data to include only years up to 2023
        df_mafindo_filtered = df_mafindo[df_mafindo['Year'] <= 2023]

        topics_per_year = df_mafindo_filtered.groupby(['Year', 'Topic']).size().reset_index(name='count')

        # Create the vertical bar chart
        bar_chart_topics = px.bar(topics_per_year, x='Year', y='count', color='Topic',
                                  color_continuous_scale=px.colors.sequential.Viridis)

        # Update layout to adjust the legend
        bar_chart_topics.update_layout(
            width=600, height=150, xaxis_title='Tahun', yaxis_title='Jumlah Topik',
            xaxis_title_font_size=10, yaxis_title_font_size=10,
            xaxis_tickfont_size=8, yaxis_tickfont_size=8, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(
                yanchor="top", y=1, xanchor="left", x=1.02,  # Adjust position of the legend
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
                bordercolor='rgba(0, 0, 0, 0)',  # No border color
                itemclick='toggleothers',  # Allow toggling of legend items
                itemsizing='constant',  # Consistent sizing for legend items
                font=dict(size=8),
                traceorder='normal',
                orientation='v',  # Vertical orientation of legend
                title_text='Topic'
            )
        )

        st.plotly_chart(bar_chart_topics, use_container_width=True)


    # Visualization 4: Horizontal Bar chart for Topics per Year using Plotly
    # with col4:
    #     st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Topics per Tahun</h6>", unsafe_allow_html=True)
    #     df_mafindo['Tanggal'] = pd.to_datetime(df_mafindo['Tanggal'], format='%d/%m/%Y')
    #     df_mafindo['Year'] = df_mafindo['Tanggal'].dt.year

    #     # Filter the data to include only years up to 2023
    #     df_mafindo_filtered = df_mafindo[df_mafindo['Year'] <= 2023]

    #     topics_per_year = df_mafindo_filtered.groupby(['Year', 'Topic']).size().reset_index(name='count')

    #     # Create the horizontal bar chart
    #     bar_chart_topics = px.bar(topics_per_year, x='count', y='Topic', color='Year', orientation='h',
    #                               color_continuous_scale=px.colors.sequential.Viridis)
    #     bar_chart_topics.update_layout(
    #         width=300, height=150, xaxis_title='Jumlah Topik', yaxis_title='Topic',
    #         xaxis_title_font_size=10, yaxis_title_font_size=10,
    #         xaxis_tickfont_size=8, yaxis_tickfont_size=8, margin=dict(t=10, b=10, l=10, r=10),
    #         showlegend=True
    #     )
    #     st.plotly_chart(bar_chart_topics, use_container_width=False)

        
    # Create a new row for WordCloud visualizations
    col5, col6, col7 = st.columns([2, 2.5, 2.5])

    # Wordcloud for Hoax
    with col5:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Wordcloud for Hoax</h6>", unsafe_allow_html=True)
        hoax_text = ' '.join(df[df['Label'] == 'HOAX']['Content'])
        wordcloud_hoax = generate_wordcloud(hoax_text, 'Reds', combined_stopwords)
        fig_hoax = plt.figure(figsize=(5, 2.5))
        plt.imshow(wordcloud_hoax, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig_hoax)
    
    # # Wordcloud for Non-Hoax
    # with col5:
    #     st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Wordcloud for Non-Hoax</h6>", unsafe_allow_html=True)
    #     non_hoax_text = ' '.join(df[df['Label'] == 'NON-HOAX']['Content'])
    #     wordcloud_non_hoax = generate_wordcloud(non_hoax_text, 'Greens', combined_stopwords)
    #     fig_non_hoax = plt.figure(figsize=(5, 2.5))
    #     plt.imshow(wordcloud_non_hoax, interpolation='bilinear')
    #     plt.axis('off')
    #     st.pyplot(fig_non_hoax)
    
    with col6:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Classification Donut Chart</h6>", unsafe_allow_html=True)
        df_classification_counts = df['Classification'].value_counts().reset_index()
        df_classification_counts.columns = ['Classification', 'Count']
    
        # Create the donut chart
        donut_chart_classification = px.pie(df_classification_counts, names='Classification', values='Count', 
                                        hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    
        # Update layout to move the legend and adjust its size
        donut_chart_classification.update_layout(
            width=300, height=170,  # Adjust the size of the chart
            margin=dict(t=20, b=20, l=20, r=120),  # Adjust margins to make room for the legend
            legend=dict(
                yanchor="top", y=1, xanchor="left", x=1.07,  # Adjust position of the legend
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
                bordercolor='rgba(0, 0, 0, 0)',  # No border color
                itemclick='toggleothers',  # Allow toggling of legend items
                itemsizing='constant',  # Consistent sizing for legend items
                font=dict(size=8),  # Smaller font size for the legend
                traceorder='normal',
                orientation='v',  # Vertical legend
                title_text='Classification'  # Title for the legend
            )
        )
        st.plotly_chart(donut_chart_classification, use_container_width=True)

    with col7:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Topic Donut Chart</h6>", unsafe_allow_html=True)
        df_mafindo_topic_counts = df_mafindo['Topic'].value_counts().reset_index()
        df_mafindo_topic_counts.columns = ['Topic', 'Count']
    
        # Create the donut chart
        donut_chart_topic = px.pie(df_mafindo_topic_counts, names='Topic', values='Count', 
                                        hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    
        # Update layout to move the legend and adjust its size
        donut_chart_topic.update_layout(
            width=250, height=170,  # Adjust the size of the chart
            margin=dict(t=20, b=20, l=20, r=100),  # Adjust margins to make room for the legend
            legend=dict(
                yanchor="top", y=1, xanchor="left", x=1.07,  # Adjust position of the legend
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background for legend
                bordercolor='rgba(0, 0, 0, 0)',  # No border color
                itemclick='toggleothers',  # Allow toggling of legend items
                itemsizing='constant',  # Consistent sizing for legend items
                font=dict(size=8),  # Smaller font size for the legend
                traceorder='normal',
                orientation='v',  # Vertical legend
                title_text='Classification'  # Title for the legend
            )
        )
        st.plotly_chart(donut_chart_topic, use_container_width=True)



        
    # Evaluation Metrics Table
    data = [
        ["indobenchmark/indobert-base-p2", 0.6898, 0.9793, 0.8094, 0.8400, 0.1981, 0.3206, 0.7023],
        ["cahya/bert-base-indonesian-522M", 0.7545, 0.8756, 0.8106, 0.6800, 0.4811, 0.5635, 0.7358],
        ["indolem/indobert-base-uncased", 0.7536, 0.8238, 0.7871, 0.6136, 0.5094, 0.5567, 0.7124],
        ["mdhugol/indonesia-bert-sentiment-classification", 0.7444, 0.8601, 0.7981, 0.6447, 0.4623, 0.5385, 0.7191]
    ]

    highest_accuracy = max(data, key=lambda x: x[-1])

    # Header Table
    html_table = """
    <table style="width:100%; border-collapse: collapse; font-size: 12px;">
        <tr>
            <th rowspan="2" style="border: 1px solid black; padding: 5px; font-size: 14px;">Pre-trained Model</th>
            <th colspan="3" style="border: 1px solid black; padding: 5px; font-size: 14px;">Label 0</th>
            <th colspan="3" style="border: 1px solid black; padding: 5px; font-size: 14px;">Label 1</th>
            <th rowspan="2" style="border: 1px solid black; padding: 5px; font-size: 14px;">Accuracy</th>
        </tr>
        <tr>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">Precision</th>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">Recall</th>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">F1-Score</th>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">Precision</th>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">Recall</th>
            <th style="border: 1px solid black; padding: 5px; font-size: 12px;">F1-Score</th>
        </tr>
    """
    # Isi Data
    for row in data:
        if row == highest_accuracy:
            html_table += "<tr style='background-color: #FFFF99; font-size: 12px;'>"
        else:
            html_table += "<tr style= ' font-size: 12px;'>"
        for item in row:
            html_table += f"<td style='border: 1px solid black; padding: 5px; font-size: 12px;'>{item}</td>"
        html_table += "</tr>"

    html_table += "</table>"
    # Tampilkan Tabel di Streamlit
    col9 = st.columns([5])
    with col9[0]:
        st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Evaluation Metrics</h6>", unsafe_allow_html=True)
        st.markdown(html_table, unsafe_allow_html=True)

    # with col9[0]:
    #     st.markdown("<h6 style='font-size: 14px; margin-bottom: 0;'>Evaluation Metrics</h6>", unsafe_allow_html=True)
    #     header = pd.MultiIndex.from_tuples([
    #         ('Pre-trained Model', ''),
    #         ('Label 0', 'Precision'),
    #         ('Label 0', 'Recall'),
    #         ('Label 0', 'F1-Score'),
    #         ('Label 1', 'Precision'),
    #         ('Label 1', 'Recall'),
    #         ('Label 1', 'F1-Score'),
    #         ('', 'Accuracy')
    #     ])

    #     data = [
    #         ["indobenchmark/indobert-base-p2", 0.6898, 0.9793, 0.8094, 0.8400, 0.1981, 0.3206, 0.7023],
    #         ["cahya/bert-base-indonesian-522M", 0.7545, 0.8756, 0.8106, 0.6800, 0.4811, 0.5635, 0.7358],
    #         ["indolem/indobert-base-uncased", 0.7536, 0.8238, 0.7871, 0.6136, 0.5094, 0.5567, 0.7124],
    #         ["mdhugol/indonesia-bert-sentiment-classification", 0.7444, 0.8601, 0.7981, 0.6447, 0.4623, 0.5385, 0.7191]
    #     ]

    #     df = pd.DataFrame(data, columns=header)
    #     st.write(df)
