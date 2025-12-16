import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from joblib import Parallel, delayed
import gc
import os
from itertools import combinations
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Script Purpose
# -----------------------------------------------------------------------------
# This script builds a Bayesian Network from dependency data, performs probabilistic
# inference, and conducts sensitivity analysis. Inference is essential because it
# allows us to compute the probability of any node (e.g., a root risk) given evidence
# about other nodes (e.g., failures or changes in other components). This is crucial
# for multi-nodal sensitivity analysis, where we want to understand how the risk
# profile of the system changes when one or more nodes are perturbed. Without
# inference, we could not quantify these conditional probabilities or assess the
# impact of combinations of node states on the overall system risk.
# -----------------------------------------------------------------------------


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


st.markdown("<p class='title'>Bayesian Network & Sensitivity Analysis Dashboard</p>", unsafe_allow_html=True)

sentence = """
The module guides users through five main steps: analysing dependencies, computing posterior probabilities, performing multi-nodal and single-point sensitivity analyses, 
building the Bayesian Network, and extracting variable elimination results. This is essential for understanding how changes in one part of the system affect overall risk.
Throughout, the module provides progress feedback, and enables users to end the module when complete.  
This makes it suitable for risk modelling, scenario analysis, and decision support in complex systems. The results are saved for further analysis and reporting in Stage 5 (E).

"""

st.markdown(f"<p class='highlight'>{sentence}</p>", unsafe_allow_html=True)


st.session_state.ended = False

if 'ended' not in st.session_state:
      st.session_state.ended = False


#st.set_page_config(page_title="RiskED - Bayesian Network Sensitivity Analysis", layout="wide")

#st.title("Bayesian Network & Sensitivity Analysis Dashboard")
#st.markdown("""
# This process performs the statistical analysis on Bayesian Network, Causal Inference, and Variable Elimination to determine the sensitivity of each component within the dependency tree.
# """)

# Step 1: Compute dependencies
st.header("Step 1: Analyse Dependencies")
progress1 = st.progress(0)
status1 = st.empty()
status1.text("Loading initial data...")


csv_file = 'data_model.csv'
if not os.path.exists(csv_file):
    st.error("The required file to perform this analysis was not found. You may have missed a step in the analysis sequence. Please ensure you have generated the dependency model before proceeding.")
    st.stop()
else:
    df = pd.read_csv('data_model.csv')



top_categories = {'Security Goal', 'People', 'Technology', 'Process'}
parent_lookup = dict(zip(df['child_value'], df['rootnode']))

def trace_category(child):
    visited = set()
    current = child
    while current not in top_categories:
        if current in visited or current not in parent_lookup:
            return None
        visited.add(current)
        current = parent_lookup[current]
    return current

df['category'] = df['child_value'].apply(trace_category)
progress1.progress(20)
status1.text("Building dependency tree...")

children_lookup = {}
for parent, child in zip(df['rootnode'], df['child_value']):
    children_lookup.setdefault(parent, []).append(child)

def count_direct_dependants(node):
    return len(children_lookup.get(node, []))

def count_cumulative_dependants(node):
    count = 0
    stack = children_lookup.get(node, []).copy()
    while stack:
        current = stack.pop()
        count += 1
        stack.extend(children_lookup.get(current, []))
    return count

df['direct_dep'] = df['child_value'].apply(count_direct_dependants)
df['cummu_dep'] = df['child_value'].apply(count_cumulative_dependants)
progress1.progress(40)
status1.text("Calculating probabilities...")

prob_lookup = dict(zip(df['child_value'], df['probability']))
dependency_lookup = {}

def compute_probability(node):
    current_prob = prob_lookup.get(node, 0.0)
    if current_prob != 0.0:
        dependency_lookup[node] = current_prob
        return current_prob
    children = children_lookup.get(node, [])
    if not children:
        dependency_lookup[node] = 0.0
        return 0.0
    product = 1.0
    for child in children:
        product *= compute_probability(child)
    dependency_lookup[node] = product
    prob_lookup[node] = product
    return product

df['dependency'] = 0.0
for idx, row in df.iterrows():
    node = row['child_value']
    if row['probability'] == 0.0 and row['dependants'] == 'Yes':
        dependency = compute_probability(node)
    else:
        dependency = row['probability']
    dependency_lookup[node] = dependency
    df.at[idx, 'dependency'] = dependency

intermediate_file = 'data_model_with_dependency.csv'
df.to_csv(intermediate_file, index=False)
progress1.progress(100)
status1.text("Dependencies computed and saved.")

# Step 2: Compute posterior probabilities
st.header("Step 2: Compute Posterior Probabilities")
progress2 = st.progress(0)
status2 = st.empty()
status2.text("Loading coefficients...")



if not os.path.exists(intermediate_file):
    st.error("❌ The required file has not been generated or was not found. You may have missed a step in the analysis sequence. Please ensure you have completed the dependency computation before proceeding.")
    st.stop()
else:
    response = pd.read_csv(intermediate_file)


coeff_file = 'data_coeff.csv'
if not os.path.exists(coeff_file):
    st.error("❌ The required file has not been generated or was not found. You may have missed a step in the analysis sequence. Please ensure you have uploaded or generated the coefficients file before proceeding.")
    st.stop()
else:
    coeff = pd.read_csv(coeff_file)

coeff['mcoefficient'] = coeff['mcoefficient'] / 100
people_sp, process_sp, tech_sp, system_sp = coeff["mcoefficient"].iloc[:4]
cp_map = {
    "People": (people_sp * system_sp, (1 - people_sp) * system_sp),
    "Process": (process_sp * system_sp, (1 - process_sp) * system_sp),
    "Technology": (tech_sp * system_sp, (1 - tech_sp) * system_sp)
}
response['posterior'] = 0.0
leaf_nodes = response["dependants"] == "No"

def calc_posterior(row):
    cp1, cp2 = cp_map[row["category"]]
    prior1 = row["probability"]
    prior2 = 1 - prior1
    xjp1 = cp1 * prior1
    xjp2 = cp2 * prior2
    xmp = xjp1 + xjp2
    return round(xjp1 / xmp, 5)

response.loc[leaf_nodes, "posterior"] = response.loc[leaf_nodes].apply(calc_posterior, axis=1)
progress2.progress(30)
status2.text("Posterior probabilities for leaf nodes calculated.")

children_lookup = {}
for parent, child in zip(response['rootnode'], response['child_value']):
    children_lookup.setdefault(parent, []).append(child)
posterior_lookup = dict(zip(response['child_value'], response['posterior']))

def compute_and_posterior(node):
    children = children_lookup.get(node, [])
    if not children:
        return posterior_lookup.get(node, 0.0)
    product = 1.0
    for child in children:
        product *= compute_and_posterior(child)
    posterior_lookup[node] = product
    return product

response['posterior_and_full'] = response['posterior']
for idx, row in response.iterrows():
    node = row['child_value']
    if row['dependants'] == 'Yes':
        and_posterior = compute_and_posterior(node)
        response.at[idx, 'posterior_and_full'] = and_posterior

security_goal_posterior = compute_and_posterior('Security Goal')
response.loc[response['child_value'] == 'Security Goal', 'posterior_and_full'] = security_goal_posterior
response.to_csv('data_model_with_posterior.csv', index=False)
progress2.progress(100)
status2.text("Posterior probabilities computed and saved.")

rootb = response.iloc[0]["rootnode"]
rootp = response.iloc[0]["posterior_and_full"]
root_dependant = 1
cum_root_dependant = response.iloc[0]["cummu_dep"]
for i in range(1, len(response)):
    if response.iloc[i]["rootnode"] == rootb:
        rootp = round(rootp * response.iloc[i]["posterior_and_full"], 9)
        root_dependant += 1
        cum_root_dependant = cum_root_dependant + response.iloc[i]["cummu_dep"]

results = {"category": "GOAL",
           "rootnode": "",
           "child_value": rootb,
           "probability": 0,
           "dependants": "Yes",
           "direct_dep": root_dependant,
           "cummu_dep": cum_root_dependant,
           "dependency": rootp,
           "posterior": rootp,
           "posterior_and_full": rootp}
response = pd.concat([response, pd.DataFrame([results])], ignore_index=True)
response.to_csv("data_model_with_posterior.csv", index=False)
progress2.progress(100)
status2.text("Root node computed and appended.")

# Step 3: Sensitivity Analysis
st.header("Step 3: Perform Sensitivity Analysis")
progress3 = st.progress(0)
status3 = st.empty()
status3.text("Computing 3-way sensitivity analysis...")

senseprob = []
sensename = [] 
sensey1 = []
sensey2 = []
mask = response["dependants"] == "No"
for idx, row in response[mask].iterrows():
    denom = row["posterior_and_full"]
    if denom != 0:
        sense3 = round((rootp / denom), 9)
        xy1 = rootp
        xy2 = sense3 - rootp
        senseprob.append(sense3)
        sensename.append(row["child_value"])
        sensey1.append(xy1)
        sensey2.append(xy2)
sensitivity = {
    "NodeName": sensename,
    "Sensitivity": senseprob,
    "y1": sensey1,
    "y2": sensey2
}
sensitivity = pd.DataFrame(sensitivity)
sensitivity.to_csv("sensitivity_p.csv", index=False)
progress3.progress(100)
status3.text("Sensitivity analysis completed.")

# Step 4: Bayesian Network & Variable Elimination
st.header("Step 4: Extract Bayesian Network & Variable Elimination")
progress4 = st.progress(0)
status4 = st.empty()
status4.text("Building Bayesian Network...")

dependency = pd.read_csv('data_model_with_posterior.csv')
edges = []
for idx, row in dependency.iterrows():
    parent = row['rootnode']
    child = row['child_value']
    if pd.notna(parent) and pd.notna(child) and parent != '' and child != '' and parent != child:
        edges.append((parent, child))
nodes = pd.unique(dependency[['rootnode', 'child_value']].values.ravel('K'))
nodes = [n for n in nodes if pd.notna(n) and n != '']

model = DiscreteBayesianNetwork(edges)
for node in nodes:
    child_rows = dependency[dependency['child_value'] == node]
    parents = child_rows['rootnode'].dropna().unique().tolist()
    if not parents:
        prob_row = dependency[dependency['child_value'] == node]
        if not prob_row.empty:
            p1 = prob_row.iloc[0]['posterior']
            p0 = 1 - p1
            cpd = TabularCPD(variable=node, variable_card=2, values=[[p0], [p1]])
            model.add_cpds(cpd)
    else:
        parent_card = [2] * len(parents)
        cpd_rows = []
        for _, row in child_rows.iterrows():
            p1 = row['posterior']
            p0 = 1 - p1
            cpd_rows.append([p0, p1])
        while len(cpd_rows) < 2 ** len(parents):
            cpd_rows.append([0.5, 0.5])
        values = list(map(list, zip(*cpd_rows)))
        cpd = TabularCPD(variable=node, variable_card=2, values=values, evidence=parents, evidence_card=parent_card)
        model.add_cpds(cpd)
assert model.check_model(), "Model is invalid!"
progress4.progress(40)
status4.text("Bayesian Network constructed.")

inference = VariableElimination(model)
root_candidates = [n for n in nodes if n not in dependency['child_value'].values]
root = root_candidates[0] if root_candidates else nodes[0]
leaf_candidates = [n for n in nodes if n not in dependency['rootnode'].values]
leaf = leaf_candidates[0] if leaf_candidates else nodes[-1]
root_result = inference.query(variables=[root])
leaf_result = inference.query(variables=[leaf], evidence={root: 1})

progress4.progress(100)
status4.text("Inference completed.")

# st.subheader(f"Marginal probability for root node '{root}':")
# st.write(root_result)
# st.subheader(f"Probability for leaf node '{leaf}' given {root}=1:")
# st.write(leaf_result)

# Step 5: Onefold Sensitivity Analysis
st.header("Step 5: Single Node Sensitivity Analysis")
progress5 = st.progress(0)
status5 = st.empty()
status5.text("Running parallel sensitivity analysis...")

# def onefold_query(node, inference, root):
#     try:
#         result = inference.query(variables=[root], evidence={node: 0})
#         prob = float(result.values[1])
#         return {'node': node, 'root_prob_given_node0': prob}
#     except Exception:
#         return {'node': node, 'root_prob_given_node0': np.nan}
    

def onefold_query(node, inference, root):
    result_dict = {'node': node}
    # What-if: node fails (0)
    try:
        result0 = inference.query(variables=[root], evidence={node: 0})
        result_dict['root_prob_given_node0'] = float(result0.values[1])
    except Exception:
        result_dict['root_prob_given_node0'] = None
    # What-if: node is perfect (1)
    try:
        result1 = inference.query(variables=[root], evidence={node: 1})
        result_dict['root_prob_given_node1'] = float(result1.values[1])
    except Exception:
        result_dict['root_prob_given_node1'] = None
    return result_dict


analysis_nodes = [node for node in nodes if node != root]
chunk_size = 1000
results = []
for i in range(0, len(analysis_nodes), chunk_size):
    chunk = analysis_nodes[i:i+chunk_size]
    chunk_results = Parallel(n_jobs=-1)(
        delayed(onefold_query)(node, inference, root) for node in chunk
    )
    results.extend(chunk_results)
    chunk_df = pd.DataFrame(chunk_results)
    mode = 'w' if i == 0 else 'a'
    header = (i == 0)
    chunk_df.to_csv('one_point_sensitivity.csv', mode=mode, header=header, index=False)
    progress5.progress(int(100 * (i + len(chunk)) / len(analysis_nodes)))
    status5.text(f"Processed {i + len(chunk)} of {len(analysis_nodes)} nodes...")
    gc.collect()
progress5.progress(100)
status5.text("Single Node sensitivity analysis completed.")

def multi_node_sensitivity_analysis_baseline(inference, nodes, root_node):
    # Exclude root, Process, Technology, People
    exclude_nodes = {root_node, "Process", "Technology", "People"}
    selectable_nodes = [n for n in nodes if n not in exclude_nodes]

    st.header("Step 6: Multi-node Sensitivity Analysis (What-If)")
    selected_nodes = st.multiselect(
        "Select up to 5 nodes for sensitivity analysis:",
        selectable_nodes,
        max_selections=5
    )

    if selected_nodes:
        n = len(selected_nodes)
        results = []
        # For each combination size (1 to n)
        for r in range(1, n+1):
            for node_subset in itertools.combinations(selected_nodes, r):
                # Only test failure (0) for selected nodes in the subset
                evidence = {node: 0 for node in node_subset}
                try:
                    result = inference.query(variables=[root_node], evidence=evidence)
                    root_prob = float(result.values[1])  # Probability root is '1'
                except Exception:
                    root_prob = None
                # Build row: all selected nodes, 1 if not in subset, 0 if in subset
                row = {f"{node} = 0": (0 if node in node_subset else 1) for node in selected_nodes}
                row["root_probability"] = root_prob
                results.append(row)
        results_df = pd.DataFrame(results)
        st.write("Sensitivity Analysis Results (What-If):")
        st.dataframe(results_df)
        results_df.to_csv("multi_node_sensitivity_results.csv", index=False)
        st.success("Results saved for final reports.")


multi_node_sensitivity_analysis_baseline(inference, nodes, root)


st.success("All analyses complete. Results saved for final report generation.")

st.markdown("---")    
if st.button("End Module"):
      st.session_state.ended = True
      if st.session_state.ended:
        st.info("The Bayesian Network & Causal Inference computation module is completed. Please return to the sidebar Menu and pick Option E to generate reports.")
        st.stop()

      #st.rerun()