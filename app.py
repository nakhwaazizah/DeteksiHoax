import streamlit as st
import pandas as pd

# Mengatur judul dan layout halaman
st.set_page_config(page_title="Dashboard", layout="wide")

# CSS untuk styling sidebar dan konten
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        padding-top: 50px;
    }
    .sidebar .sidebar-content a {
        font-size: 18px;
        color: #555;
        padding: 10px;
        text-decoration: none;
    }
    .sidebar .sidebar-content a:hover {
        background-color: #e6ecf0;
        border-radius: 5px;
    }
    .sidebar .sidebar-content .active {
        background-color: #fff;
        border-left: 5px solid #2e7bcf;
        border-radius: 5px;
    }
    .css-1aumxhk {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigasi
pages = {
    "Home": "Home",
    "Hoax Detection": "Hoax Detection",
    "Upload Data": "Upload Data",
    "Model": "Model"
}

# Menambahkan sidebar dengan link navigasi
st.sidebar.title("Dashboard")
selection = st.sidebar.radio("", list(pages.keys()))

# Home Page
if selection == "Home":
    st.title("Home")
    st.write("Welcome to the Home Page!")
    st.subheader("Data")
    data = {
        "Name": ["Home Decor Range", "Disney Princess Pink Bag 18'", "Bathroom Essentials", "Apple Smartwatches"],
        "Popularity": [45, 29, 18, 25],
        "Sales": ["45%", "29%", "18%", "25%"]
    }
    
    df = pd.DataFrame(data)
    st.table(df)

# Hoax Detection Page
elif selection == "Hoax Detection":
    st.title("Hoax Detection")
    st.write("This is the Hoax Detection page.")

# Upload Data Page
elif selection == "Upload Data":
    st.title("Upload Data")
    st.write("This is the Upload Data page.")

# Model Page
elif selection == "Model":
    st.title("Model")
    st.write("This is the Model page.")

# Menandai halaman yang dipilih sebagai aktif di sidebar
st.markdown(
    f"""
    <script>
    const pages = {list(pages.keys())};
    let links = window.parent.document.querySelectorAll('.sidebar-content a');
    links.forEach((link, i) => {{
        if (link.innerText === "{selection}") {{
            link.classList.add('active');
        }}
    }});
    </script>
    """,
    unsafe_allow_html=True
)
