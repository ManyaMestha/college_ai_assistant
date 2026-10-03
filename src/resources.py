from pathlib import Path


# --------------------------------------------------
# 1. Main resource folder
# --------------------------------------------------

RESOURCE_FOLDER = Path("documents")


# --------------------------------------------------
# 2. Get available subjects
# --------------------------------------------------

def get_subjects():

    subjects = []

    for folder in RESOURCE_FOLDER.iterdir():

        if folder.is_dir() and folder.name != "ACADEMICS":
            subjects.append(folder.name)

    return sorted(subjects)


# --------------------------------------------------
# 3. Get categories for a subject
# --------------------------------------------------

def get_categories(subject):

    subject_folder = RESOURCE_FOLDER / subject

    categories = []

    if subject_folder.exists():

        for folder in subject_folder.iterdir():

            if folder.is_dir():
                categories.append(folder.name)

    return sorted(categories)


# --------------------------------------------------
# 4. Get PDFs for a category
# --------------------------------------------------

def get_pdfs(subject, category):

    category_folder = RESOURCE_FOLDER / subject / category

    pdfs = []

    if category_folder.exists():

        for pdf_file in category_folder.glob("*.pdf"):

            pdfs.append({
                "name": pdf_file.name,
                "path": str(pdf_file)
            })

    return pdfs