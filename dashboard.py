import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data.csv")  # Update the path if needed

# Convert 'Tanggal' to datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')

# Number of records per datasource
datasource_counts = df["Datasource"].value_counts()
fig_datasource = px.bar(datasource_counts, x=datasource_counts.index, y=datasource_counts.values, title="Number of Records per Datasource")
fig_datasource.update_layout(height=250, width=250)

# Number of records per label
label_counts = df["Label"].value_counts()
fig_label_pie = px.pie(values=label_counts.values, names=label_counts.index, title="Number of Records per Label")
fig_label_pie.update_layout(height=300, width=300)  # Increased size for pie chart

# Hoax and Non-Hoax per year
df['Year'] = df['Tanggal'].dt.year
yearly_counts = df.groupby(['Year', 'Label']).size().unstack(fill_value=0)
fig_yearly = px.line(yearly_counts, x=yearly_counts.index, y=['HOAX', 'NON-HOAX'], title='Hoax and Non-Hoax per Year')
fig_yearly.update_layout(height=250, width=250)

# Word Cloud
text = " ".join(content for content in df.Content)
wordcloud = WordCloud(width=300, height=150, background_color='white').generate(text)
    
fig_wordcloud, ax = plt.subplots(figsize=(3, 1.5), dpi=100)
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
plt.tight_layout(pad=0)  # Remove extra space around the wordcloud

# Arrange visualizations into two per row
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_datasource, use_container_width=True)
    st.plotly_chart(fig_label_pie, use_container_width=True)

with col2:
    st.plotly_chart(fig_yearly, use_container_width=True)
    st.subheader("Word Cloud of Content")
    st.pyplot(fig_wordcloud)  # Display WordCloud only once
