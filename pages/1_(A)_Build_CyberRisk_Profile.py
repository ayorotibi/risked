import streamlit as st
import pandas as pd
import statistics

# This module reads standard security profile statements from a CSV file (Security_Profie.csv) and 
# allows users to respond to them. The response is saved in Security_Profile_Responses.csv.
# The responses are then analysed to compute category coefficients which are saved in data_coeff.csv.
# In addition, the csv file will be submitted to genAI with a quiry to generate Cybersecurity Status/Profile report for the user

# Color coding  red color = #FF5733;    blue color:#2E86C1 black color: #000000;   green color: #28B463

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
st.session_state.ended = False

st.markdown("<p class='title'>Cyber Risk Profiler</p>", unsafe_allow_html=True)

sentence = """
This module accepts responses from user to determine their current Cyber Risk PROFILE of the system.
The user is presented with 15 standard risk profile statements, based on 12 widely used ICS Security Frameworks and Standards
and they must select their response from a dropdown menu.
The responses shall be used to provide you with your Security Profile, based on the existing Security Frameworks. (NOTE: There are over 100 statements in the full version of the tool,
and 40 will be dynamically selected based on SECTOR).
"""

st.markdown(f"<p class='highlight'>{sentence}</p>", unsafe_allow_html=True)




# --- Load statements ---
uploaded_file = 'Security_Profile_15.csv'
if uploaded_file:
    df = pd.read_csv(uploaded_file)

response_options = ["Strongly Agreed", "Agreed", "Disagreed", "Strongly Disagreed", "Not Applicable"]
evidence_options = ["yes", "no"]

# --- Initialise session state ---
if 'responses' not in st.session_state:
    st.session_state.responses = [{} for _ in range(len(df))]
if 'saved' not in st.session_state:
    st.session_state.saved = False
if 'ended' not in st.session_state:
    st.session_state.ended = False



# --- Step 1: Input responses ---
st.header("Step 1: Please, respond to each statement")
for index, row in df.iterrows():
    #st.markdown(f"**{index+1}: {row['Security Profile Statement']}**")
    statements = (f"*{index + 1}: {row['Security Profile Statement']}")
    st.markdown(f"<p class='subtitle'>{statements}</p>", unsafe_allow_html=True)
    response = st.selectbox(
        "Select RESPONSE to Statement",
        response_options,
        key=f"response_{index}",
        index=response_options.index(st.session_state.responses[index].get("Response", response_options[0]))
    )
    evidence = st.selectbox(
        "Can you EVIDENCE your response?",
        evidence_options,
        key=f"evidence_{index}",
        index=evidence_options.index(st.session_state.responses[index].get("Evidence", evidence_options[0]))
    )
    # Store in session_state
    st.session_state.responses[index] = {
        "Category": row["Category"],
        "Security Profile Statement": row["Security Profile Statement"],
        "Weight": row["Weight"],
        "Response": response,
        "Evidence": evidence
    }

# --- Step 2: Review & Save ---
st.header("Step 2: Review & Save your responses")
st.write("You can review and change your responses above. When ready, click 'Save Responses'.")

if st.button("Save Responses"):
    response_df = pd.DataFrame(st.session_state.responses)
    response_df.to_csv("Security_Profile_Responses.csv", index=False)
    st.session_state.saved = True
    st.success("Responses saved successfully! Please scroll down for more information")

# --- Step 3: Show analysis and End button only after saving ---
if st.session_state.saved:
    #st.write("Here are your responses:")
    st.markdown("<p class='subtitle'>Here are your responses:</p>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.responses))

    # --- Analysis function ---
    def analyse():
        data = pd.read_csv("Security_Profile_Responses.csv")
        data = data.sort_values(by=['Category'])
        category_scores = {"People": [], "Process": [], "Technology": [], "System": []}
        likert_map = {
            "Strongly Agreed": 4.0,
            "Agreed": 3.0,
            "Disagreed": 2.0,
            "Strongly Disagreed": 1.0
        }
        for idx, row in data.iterrows():
            response = row["Response"]
            category = row["Category"]
            weight = row["Weight"]
            evidence = row["Evidence"]
            if response == "Not Applicable" or category not in category_scores:
                continue
            score = likert_map.get(response)
            if score is None:
                continue
            if evidence == "no" and response in ["Strongly Agreed", "Agreed"]:
                score *= 0.5
            score *= weight
            category_scores[category].append(score)
        coeff_data = {"mcategory": [], "mcoefficient": []}
        for cat, scores in category_scores.items():
            mean_score = statistics.mean(scores) if scores else 0
            coeff_data["mcategory"].append(cat)
            coeff_data["mcoefficient"].append(mean_score)
        df = pd.DataFrame(coeff_data)
        df.to_csv("data_coeff.csv", index=False)
        #st.write("Here is the Security Profile Coefficient for your system")
        st.markdown("<p class='subtitle'>Here is the Security Profile Coefficient (SPC) for your system</p>", unsafe_allow_html=True)
        st.dataframe(df)

    analyse()

    if st.button("End Module"):
        st.session_state.ended = True
        st.info("The Cyber Risk Profiling module is completed. Return to the sidebar for next module.")
        st.stop()
