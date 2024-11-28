import os
import json
import streamlit as st
import networkx as nx
import pickle
import matplotlib.pyplot as plt

def find_files(base_dir, file_extension):
    files_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(file_extension):
                files_list.append(os.path.join(root, file))
    return files_list

def display_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    st.json(data)

def display_graph(file_path):
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, with_labels=True, ax=ax, node_color='skyblue', node_size=100, edge_color='gray', font_size=8)
    st.pyplot(fig)

def display_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        log_data = file.read()
    st.text(log_data)

def main():
    st.title("File Viewer")

    # User input for directory
    base_dir = st.text_input("Enter the base directory (e.g., ./graphs)")
    if not base_dir:
        st.warning("Please enter a valid base directory.")
        return

    # File type selector
    file_type = st.selectbox("Select file type", ["JSON", "Pickle", "Log"])

    # List files based on type
    file_extension = ".json" if file_type == "JSON" else (".pkl" if file_type == "Pickle" else ".log")
    files = find_files(base_dir, file_extension)

    if not files:
        st.warning(f"No {file_type} files found in the specified directory.")
        return

    selected_file = st.selectbox(f"Choose a {file_type} file", files)

    # Display file based on selected type
    if selected_file:
        st.write(f"Displaying: {selected_file}")
        if file_type == "JSON":
            display_json_file(selected_file)
        elif file_type == "Pickle":
            display_graph(selected_file)
        else:
            display_log_file(selected_file)

if __name__ == "__main__":
    main()
