import streamlit as st
from text.loaders import DocumentLoader
from text.utils import clean_text

# Title and description
st.title("StudyBot")
st.write("Upload your documents, select your options, and generate Anki cards!")

# File upload
uploaded_files = st.file_uploader(
    "Upload Files", type=["pdf", "docx", "txt", "srt"], accept_multiple_files=True
)

# Document selection
selected_files = st.multiselect(
    "Select Documents", [file.name for file in uploaded_files]
)

if uploaded_files:
    st.write("Documents Preview:")
    for uploaded_file in uploaded_files:
        if uploaded_file.name in selected_files:
            # store the document temporarily
            temp_file_path = f"/tmp/{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                documents = DocumentLoader.load_documents(temp_file_path)

            except ValueError as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")

# Customization options
num_cards = st.slider(
    "Number of Cards to Generate", min_value=1, max_value=50, value=10
)
difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])

# Generate button
if st.button("Generate Cards"):
    st.write(f"Generating {num_cards} cards with {difficulty} difficulty...")
    # Call backend logic for card generation here

# Display generated cards (placeholder for now)
if uploaded_files and selected_files:
    st.write("Generated Cards Preview:")
    st.text_area("Preview", "Multiple-choice questions will be shown here.")

# Export section
st.download_button(
    "Download .apkg File", data="Generated Anki Cards", file_name="anki_cards.apkg"
)
