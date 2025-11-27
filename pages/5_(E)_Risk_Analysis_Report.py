import streamlit as st
import os
import requests
import docx
from datetime import datetime
import io

st.session_state.ended = False

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


st.markdown("<p class='title'>Risk Analysis Report Generator</p>", unsafe_allow_html=True)

sentence = """
This module generates a comprehensive Cyber Risk and Resiliency Report for industrial systems, using the results of a prior sensitivity analysis. 
 It prompts the user for an API key and business sector. The tool constructs a detailed prompt and sends it to an AI model to produce a report tailored for senior leadership. 
 The generated report is displayed in the tool and can be downloaded as a formatted Word document. 
 The report covers executive summary, risk analysis, sensitivity outcomes, and actionable recommendations, all based on the system’s dependency structure and risk metrics. 
"""

st.markdown(f"<p class='highlight'>{sentence}</p>", unsafe_allow_html=True)


if 'ended' not in st.session_state:
      st.session_state.ended = False

if st.session_state.ended:
      st.info("The Module produced the Reports. It is the last action to be performed. Please close this tab and return to the RISKED Menu tab.")
      st.stop()


# --- CONFIGURATION ---
DEFAULT_MODEL = "llama-3.1-8b-instant"
MODEL = DEFAULT_MODEL
required_files = [
    "one_point_sensitivity.csv",
    "data_model_with_posterior.csv"
]
# Hardcoded values
sector = "Manufacturing"


# --- Ask user for API key in sidebar ---
st.sidebar.header("Groq/Llama API Key")
api_key = st.sidebar.text_input(
    "Enter your Groq/Llama API Key:",
    type="password",
    help="Paste your Groq/Llama API key here. It will be used for all API requests."
)

if not api_key:
    st.warning("Please enter your Groq/Llama API key in the sidebar to continue.")
    st.stop()


# In the sidebar, ask for company name
sector = st.sidebar.text_input(
    "Enter your organisation's Sector:",
    help="Type your organisation's Sector so as to focus the analysis."
)

if not sector:
    st.warning("Please enter your organisation's Sector in the sidebar to continue.")
    st.stop()


PROMPT_TEMPLATE = """
Business Sector: {sector}
You are an AI expert in cybersecurity risk analysis for complex industrial systems like SCADA and ICS. Your specialty is using probabilistic models to understand how failures and attacks can propagate through interdependent components. You have encyclopedic knowledge of relevant standards, including MITRE ATT&CK for ICS, NIST SP 800-82, ISA/IEC 62443, and others listed.
You are provided with two CSV files:
- one_point_sensitivity.csv
- data_model_with_posterior.csv
Generate a comprehensive Cyber Risk and Resiliency Report based on an analysis of two provided CSV files: one_point_sensitivity.csv and data_model_with_posterior.csv. The report must be accessible to a non-technical senior leadership audience (e.g., CISO, CRO, Asset Owners) while being analytically rigorous.
Try to use your own word, rathar than copying large chunks of the prompt. Avoild repeating the same phrases and words used as examples in the prompt.
Step-by-Step Instructions:
Step 1: Data Interpretation & Setup
The primary data for sensitivity analysis is in the one_point_sensitivity.csv file.
Identify the prob_given_node0 column. This represents the Probability Sensitivity Score. This score states that if the given node (leaf node) were to be made perfectly unreliable (0% chance of sucess), the score indicates the new marginal (success) probability of the root node.
Interpretation Rule: A lower percentage in the prob_given_node0 column indicates a more critical node. This is how faw away from 100%. Therefore, treat this as a core metric for your analysis.
Use the data_model_with_posterior.csv to understand the system's structure, dependencies, and the baseline probabilities (e.g., the marginal probability of the root node).
When reporting any probability or sensitivity score, always format it as a percentage with two decimal places (e.g., report 0.2134300300 as 21.34%).
Step 2: Generate the Report - Structure and Content
Structure your report exactly into the following four parts:
Part 1: Executive Summary
System Structure Overview: Provide a high-level, non-technical description of the system based on the dependency model. Explain what the "root node" represents (e.g., "Overall System Availability" or "Core Production Process").
Overall System Risk: State the current marginal probability of the root node. Explain what this probability means in business terms (e.g., "a 15% probability of a major operational disruption").
System Characteristics: Briefly mention the dependency depth, interactive complexity, and coupling among nodes, explaining what these mean for system stability and risk management.
Key Risk Points: Identify the most critical leaf node(s) from your sensitivity analysis and summarize their potential impact on the overall system's performance.
Part 2: Detailed Probabilistic Risk Analysis
Use clear, professional language to answer these three classic risk analysis questions. Assume the reader is a manager, not an engineer.
What can go wrong? Based on the dependency model and sensitivity scores, define 2-3 plausible failure scenarios. Describe them in terms of business operations (e.g., "A failure in the 'Historian Database' node could lead to a loss of process visibility for operators").
How likely is it to go wrong? For each scenario, provide a qualitative assessment (e.g., "Moderately Likely") supported by the quantitative data from the model (mention the relevant probabilities from the CSV files).
What are the consequences? For each scenario, evaluate the potential business, safety, regulatory, and financial impacts. Connect these consequences to the frameworks you know (e.g., "This scenario aligns with the MITRE ATT&CK technique T880: Loss of View and could lead to non-compliance with NERC CIP standards").
Part 3: Sensitivity Analysis Outcomes
This section should be data-driven and prescriptive.
Top 5 Critical Nodes: List the top 5 most critical leaf nodes, ranked by their Probability Sensitivity Score (remember: lower percentage = more critical).
Impact Analysis: For each of these 5 nodes, explain:
How a failure in that node would impact the probability of the root node failing.
How the root node's risk would improve (i.e., its failure probability would decrease) if that specific leaf node were guaranteed to be 100% reliable.
Part 4: Recommendations & Strategic Reminders
Targeted Mitigation Strategies: Provide specific, actionable recommendations to address the top 5 critical nodes identified. Your suggestions should be based on the system's state and the sensitivity relationships. For example, "For the highly sensitive 'PLC Controller A,' implement application whitelisting as per ISA-62443-3-3."
Standard Security Hygiene: Include two concluding paragraphs on foundational security practices. Frame these as essential, non-negotiable basics. Base this reminder on your knowledge of the relevant framework, standards and regulations.
Here are the CSV file contents:
--- one_point_sensitivity.csv ---
{one_point_sensitivity}
--- data_model_with_posterior.csv ---
{data_model_with_posterior}
"""

def read_file_as_text(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def call_groq(prompt, api_key, model):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI expert in cybersecurity risk analysis for complex industrial systems like SCADA and ICS. "
                    "Your specialty is using probabilistic models to understand how failures and attacks can propagate through interdependent components. "
                    "Provide detailed, actionable analysis and recommendations. Be comprehensive and professional."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6,
        "max_tokens": 4000
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def generate_word_report(report, sector):
    doc = docx.Document()
    doc.add_heading('Executive Risk & Resiliency Report', 0)
    doc.add_paragraph(f"Business Sector: {sector}")
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")
    lines = report.split('\n')
    for line in lines:
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.strip() == "---":
            doc.add_paragraph("")
        elif line.strip().startswith("**") and line.strip().endswith("**"):
            p = doc.add_paragraph()
            p.add_run(line.strip()[2:-2]).bold = True
        elif line.strip():
            doc.add_paragraph(line)
        else:
            doc.add_paragraph("")
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

if st.button("Generate and Download Report"):
    # Step 1: Check files
    missing_files = [f for f in required_files if not os.path.isfile(f)]
    
    if missing_files:
        st.error(f"The required file(s) to perform this process are missing."
        "You may have missed a step in the analysis sequence. "
        "Please complete the previous analysis steps to generate these files before proceeding.")
        st.stop()
    
    else:
        file_texts = {filename: read_file_as_text(filename) or "" for filename in required_files}
        prompt = PROMPT_TEMPLATE.format(
            sector=sector,
            one_point_sensitivity=file_texts["one_point_sensitivity.csv"],
            data_model_with_posterior=file_texts["data_model_with_posterior.csv"]
        )
        report = call_groq(prompt, api_key, MODEL)
        if report.startswith("API Error") or report.startswith("Unexpected Error"):
            st.error(f"Please provide a valid API key to proceed. You may go to Groq/Llama to generate a free API Key, if you do not have one.")
            st.stop()
            #st.error(report)
        else:
            # Display the report on the screen
            st.subheader("Generated Risk Analysis Report")
            st.markdown(report)
            # Provide download button for Word doc
            word_buffer = generate_word_report(report, sector)
            st.download_button(
                label="Download Risk Analysis Report (Word)",
                data=word_buffer,
                file_name=f"Risk_Resiliency_Report_{sector.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
st.caption("This is a PoC and it is powered by Groq and Llama 3. In the full version, this is powered by our purpose-built genAI for Cyber risk identification and assessement.")
st.markdown("---")    
if st.button("End Module"):
      st.session_state.ended = True
      st.rerun()