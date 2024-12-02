import streamlit as st


load_data_page = st.Page(
    "app_dashboard/load_data.py",
    title="Load Data",
    icon=":material/data_usage:",
)
keyword_search_page = st.Page(
    "app_filters/keyword_search.py",
    title="Keyword Search",
    icon=":material/find_in_page:",
)
semantic_search_page = st.Page(
    "app_filters/semantic_search.py",
    title="Semantic Search",
    icon=":material/search_insights:",
)
dashboard_page = st.Page(
    "app_dashboard/dashboard.py",
    title="Dashboard",
    icon=":material/dashboard:",
)


pg = st.navigation(
    [load_data_page, keyword_search_page, semantic_search_page, dashboard_page]
)
# set layout of streamlit app
st.set_page_config(layout="wide", page_title="Job Postings Analysis")
pg.run()
