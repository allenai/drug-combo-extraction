import streamlit_single_relation_app
import streamlit_all_relations_app
import streamlit as st
PAGES = {
    "Extract All Relations": streamlit_all_relations_app,
    "Classify Single Relation": streamlit_single_relation_app
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()