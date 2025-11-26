import streamlit as st
import pandas as pd
import os
from graphviz import Digraph

st.session_state.ended = False
# if 'ended' not in st.session_state:
#       st.session_state.ended = False

if st.session_state.ended:
      st.info("The Model Mapping module is completed.")
      #st.stop()

# --- The rest of your app code goes below ---
# (All widgets, forms, tables, etc.)

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size:30px !important;
        #color:#FF5733;
        color:#2E86C1;     
        text-align:center;
    }
    .subtitle {
        font-size:20px !important;
        color:#FF5733;
        text-align:left;
    }
    .highlight {
        color:#28B463;  
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<p class='title'>System Dependency Model Builder</p>", unsafe_allow_html=True)

sentence = """
This Module allows users to build a System Dependency Model by defining root and child nodes.
The dependency tree is connected by relationships among components, based on information provided by the user.
This POC is limited to 30 nodes. You will be able to downloasd your system model graph as a PNG image at the end of the module.
"""

st.markdown(f"<p class='highlight'>{sentence}</p>", unsafe_allow_html=True)

# Configuration
csv_file = "data_model.csv"
MAX_NODES = 30
PROTECTED_NODES = ["People", "Technology", "Process"]

@st.cache_data
def load_rootnode_values():
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            required_columns = ["rootnode", "child_value", "dependants", "probability"]
            if not all(col in df.columns for col in required_columns):
                st.error("Invalid CSV format. Required columns: rootnode, child_value, dependants, probability")
                return []
            return df['rootnode'].dropna().unique().tolist()
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return []
    return []

def visualize_dependencies(df):
    dot = Digraph(comment="Dependency Tree")
    dot.attr(rankdir='LR')  # Left to right orientation

    # Fixed node size parameters
    node_width = "0.8"
    node_height = "0.4"

    if not df.empty:
        root_nodes = set(df['rootnode'])
        child_nodes = set(df['child_value'])
        leaf_nodes = {row['child_value'] for _, row in df.iterrows() if row['dependants'] == "No"}

        # Draw root nodes as green boxes
        for node in root_nodes:
            dot.node(
                node,
                label=node,
                shape='box',
                style='filled',
                fillcolor='green',
                width="1.2",
                height=node_height,
                fixedsize='true'
            )
        # Draw child nodes
        for node in child_nodes:
            if node in leaf_nodes:
                dot.node(
                    node,
                    label=node[:8],
                    shape='circle',
                    style='filled',
                    fillcolor='tomato',
                    width=node_width,
                    height=node_height,
                    fixedsize='true'
                )
            else:
                dot.node(
                    node,
                    label=node[:8],
                    shape='box',
                    style='filled',
                    fillcolor='lightgreen',
                    width=node_width,
                    height=node_height,
                    fixedsize='true'
                )

        # Draw edges
        for _, row in df.iterrows():
            dot.edge(
                row['rootnode'],
                row['child_value'],
                label=f"{row['probability']}" if row['dependants'] == "No" else ""
            )
        st.graphviz_chart(dot)
    return dot

def save_graph_image(dot, filename="dependency_graph"):
    dot.format = 'png'
    dot.render(filename, cleanup=True)

if 'rootnode_values' not in st.session_state:
    initial_values = ["Technology", "People", "Process"]
    st.session_state.rootnode_values = initial_values + load_rootnode_values()

if 'child_value' not in st.session_state:
    st.session_state.child_value = ""
if 'dependants' not in st.session_state:
    st.session_state.dependants = True
if 'probability' not in st.session_state:
    st.session_state.probability = 0.0

try:
    df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
    st.subheader("Node Dependency Records")
    st.dataframe(df)
    #dot = visualize_dependencies(df)
    dot = Digraph(comment="System (Dependency) Tree")
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    df = pd.DataFrame()
    dot = Digraph(comment="System (Dependency) Tree")


# --- Recursive delete function ---
def delete_with_dependants(df, parent_node):
    # Find all direct children of the parent_node
    children = df[df['rootnode'] == parent_node]['child_value'].tolist()
    # Remove the parent_node itself
    df = df[df['child_value'] != parent_node]
    # Recursively delete all children and their dependants
    for child in children:
        df = delete_with_dependants(df, child)
    return df

# Initialise session state
if 'rootnode_values' not in st.session_state:
    st.session_state.rootnode_values = ["Technology", "People", "Process"] + load_rootnode_values()
if 'child_value' not in st.session_state:
    st.session_state.child_value = ""
if 'dependants' not in st.session_state:
    st.session_state.dependants = "Yes"
if 'probability' not in st.session_state:
    st.session_state.probability = 0.50
if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None

# Main Interface
st.subheader("Node Dependency Record")

# Data Display and Edit/Delete Controls
try:
    df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
    with st.expander("Current Nodes"):
        if not df.empty:
            for idx, row in df.iterrows():
                cols = st.columns([2, 2, 2, 2, 1, 1])
                cols[0].write(row['rootnode'])
                cols[1].write(row['child_value'])
                cols[2].write(row['dependants'])
                cols[3].write(f"{row['probability']:.2f}")
                # Disable edit/delete for special nodes
                if row['child_value'] in PROTECTED_NODES:
                    cols[4].button("Edit", key=f"edit_{idx}", disabled=True)
                    cols[5].button("Delete", key=f"delete_{idx}", disabled=True)
                else:
                    if cols[4].button("Edit", key=f"edit_{idx}"):
                        st.session_state.child_value = row['child_value']
                        st.session_state.dependants = row['dependants']
                        st.session_state.probability = row['probability']
                        st.session_state.rootnode_selected = row['rootnode']
                        st.session_state.edit_index = idx
                    if cols[5].button("Delete", key=f"delete_{idx}"):
                        # --- Cascading delete ---
                        node_to_delete = row['child_value']
                        df = delete_with_dependants(df, node_to_delete).reset_index(drop=True)
                        df.to_csv(csv_file, index=False)
                        st.success(f"Node '{node_to_delete}' and all its dependants were deleted.")
                        st.rerun()
        else:
            st.info("No nodes in the model yet.")
    st.subheader("System Dependency Model Visualization")
    visualize_dependencies(df)
except Exception as e:
    st.error(f"Data loading error: {str(e)}")

# Input Form (Add or Edit)
with st.form("node_form"):
    # Filter out "Security Goal" from root node dropdown
    filtered_rootnodes = [node for node in st.session_state.rootnode_values if node != "Security Goal"]
    # If editing, use the selected rootnode, else default to selectbox
    if 'rootnode_selected' in st.session_state and st.session_state.edit_index is not None:
        # If the selected rootnode is "Security Goal", fallback to first available
        if st.session_state.rootnode_selected == "Security Goal":
            rootnode = st.selectbox("Root Node", filtered_rootnodes)
        else:
            rootnode = st.selectbox("Root Node", filtered_rootnodes, index=filtered_rootnodes.index(st.session_state.rootnode_selected))
    else:
        rootnode = st.selectbox("Root Node", filtered_rootnodes)
    child_value = st.text_input("Child Node", st.session_state.child_value)
    dependants = st.radio("Has Dependants?", ["Yes", "No"], index=0 if st.session_state.dependants == "Yes" else 1)

    # Probability input logic (slider)
    if dependants == "Yes":
        st.session_state.probability = 0.0
        st.slider(
            "Probability",
            min_value=0.00,
            max_value=1.00,
            value=0.0,
            step=0.01,
            key="probability_slider",
            disabled=True
        )
        probability = 0.0
    else:
        probability = st.slider(
            "Probability",
            min_value=0.00,
            max_value=1.00,
            value=st.session_state.get("probability", 0.50),
            step=0.01,
            key="probability_slider"
        )
        st.session_state.probability = probability

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.edit_index is not None:
            submit_label = "Update"
        else:
            submit_label = "Save"
        submitted = st.form_submit_button(submit_label)
    with col2:
        if st.form_submit_button("Clear"):
            st.session_state.child_value = ""
            st.session_state.dependants = "No"
            st.session_state.probability = 0.50
            st.session_state.edit_index = None
            if 'rootnode_selected' in st.session_state:
                del st.session_state.rootnode_selected
            st.rerun()
    with col3:
        if st.session_state.edit_index is not None:
            if st.form_submit_button("Cancel Edit"):
                st.session_state.child_value = ""
                st.session_state.dependants = "No"
                st.session_state.probability = 0.50
                st.session_state.edit_index = None
                if 'rootnode_selected' in st.session_state:
                    del st.session_state.rootnode_selected
                st.rerun()

# Custom error messages and Save/Update Logic
if 'submitted' in locals() and submitted:
    # Custom error messages for user input
    if not child_value.strip():
        st.error("❌ Please enter a name for the child node. This field cannot be left blank.")
    elif not rootnode:
        st.error("❌ Please select a root node from the list.")
    elif dependants not in ["Yes", "No"]:
        st.error("❌ Please indicate whether this node has dependants.")
    elif not (0.0 <= probability <= 1.0):
        st.error("❌ Probability must be between 0.00 and 1.00. Please adjust the slider.")
    elif st.session_state.edit_index is None and not df.empty and child_value in df['child_value'].values:
        st.error(f"❌ A node named '{child_value}' already exists. Please choose a unique name for each node.")
    elif st.session_state.edit_index is None and len(df) >= MAX_NODES:
        st.warning(f"⚠️ Sorry, you have reached the maximum of {MAX_NODES} nodes for this proof of concept. Please remove a node before adding a new one.")
    else:
        try:
            if st.session_state.edit_index is not None:
                # Update existing node
                df.at[st.session_state.edit_index, "rootnode"] = rootnode
                df.at[st.session_state.edit_index, "child_value"] = child_value
                df.at[st.session_state.edit_index, "dependants"] = dependants
                df.at[st.session_state.edit_index, "probability"] = probability if dependants == "No" else 0.0
                df.to_csv(csv_file, index=False)
                st.success("✅ Node updated successfully!")
                st.session_state.child_value = ""
                st.session_state.dependants = "No"
                st.session_state.probability = 0.50
                st.session_state.edit_index = None
                if 'rootnode_selected' in st.session_state:
                    del st.session_state.rootnode_selected
                st.rerun()
            else:
                # Add new node
                new_data = {
                    "rootnode": rootnode,
                    "child_value": child_value,
                    "dependants": dependants,
                    "probability": probability if dependants == "No" else 0.0
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                df.to_csv(csv_file, index=False)
                if dependants == "Yes" and child_value not in st.session_state.rootnode_values:
                    st.session_state.rootnode_values.append(child_value)
                st.success("✅ Node saved successfully!")
                st.session_state.child_value = ""
                st.session_state.dependants = "No"
                st.session_state.probability = 0.50
                st.rerun()
        except Exception as e:
            st.error(f"❌ Save failed: {str(e)}")

# Module termination
st.markdown("---")
if st.button("Click to Generate the System Model Graph"):
    save_graph_image(dot, "system_model_graph")
    #st.success("Dependency graph saved as system_model_graph.png!")
    st.session_state.show_download = True

# Show download button if requested
if st.session_state.get("show_download", False):
    if os.path.exists("system_model_graph.png"):
        with open("system_model_graph.png", "rb") as img_file:
            st.download_button(
                label="Download System Model Graph",
                data=img_file,
                file_name="system_model_graph.png",
                mime="image/png"
            )
    # Show a finish button to end the module
    if st.button("Finish"):
        st.session_state.ended = True
        st.session_state.show_download = False
        st.rerun()

# if st.session_state.ended:
#     st.info("Module ended. You can close this tab or stop the Streamlit server.")
