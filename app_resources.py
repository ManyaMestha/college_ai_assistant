import streamlit as st

from src.resources import get_subjects
from src.resources import get_categories
from src.resources import get_pdfs


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="College Study Materials",
    page_icon="📚"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 College Study Materials")

st.write(
    "Select a subject and material type to access "
    "the available PDF resources."
)


# --------------------------------------------------
# 1. Select subject
# --------------------------------------------------

subjects = get_subjects()

selected_subject = st.selectbox(
    "Select Subject",
    subjects
)


# --------------------------------------------------
# 2. Select material category
# --------------------------------------------------

categories = get_categories(selected_subject)

selected_category = st.selectbox(
    "Select Material Type",
    categories
)


# --------------------------------------------------
# 3. Get available PDFs
# --------------------------------------------------

pdfs = get_pdfs(
    selected_subject,
    selected_category
)


# --------------------------------------------------
# 4. Display PDFs
# --------------------------------------------------

st.subheader("Available PDFs")


if not pdfs:

    st.info(
        "No PDF files are currently available "
        "in this category."
    )

else:

    for pdf in pdfs:

        st.write(f"📄 {pdf['name']}")

        with open(pdf["path"], "rb") as file:

            pdf_data = file.read()

        st.download_button(
            label=f"Download {pdf['name']}",
            data=pdf_data,
            file_name=pdf["name"],
            mime="application/pdf"
        )