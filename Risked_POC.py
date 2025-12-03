
import streamlit as st

# Set page configuration (only in main script)
st.set_page_config(
    page_title="RiskED - System Modelling",
    layout="centered"
)


# Sidebar YouTube link
youtube_url = "https://youtu.be/wZ1Z-zc2X3c"   # Replace with your actual video link
st.sidebar.markdown(
    f'<a href="{youtube_url}" target="_blank">▶️ Click to watch an Intro video on YouTube</a>',
    unsafe_allow_html=True
)


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
        font-size:25px !important;
        color:#FF5733;
        text-align:center;
        font-weight:bold;
    }
    .highlight {
        font-size:18px !important;
        color:#2E86C1;  
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='title'>RiskED - Cyber Risk Identification & Assessment Tool</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>This is a Proof of Concept (PoC)</p>", unsafe_allow_html=True)

# st.title("RiskED - Cyber Risk Identification & Assessment Tool")
# st.header("Welcome to the Demo (Proof of Concept)")

sentence1 = """
"If you know the enemy and know yourself; in a hundred battles, you will never be defeated" - Sun Tzu.
"""

sentence2 = """
This Tool provides a left-shift methodology to knowing yourself (System). It is built for Complex systems such as Industrial Control Systes (ICS), Oil & Gas,
Manufacturing process, Transportation, etc. to identify and assess cyber risks in a system.
"""


sentence3 = """
Please, use the sidebar to select a process to run. 
For best results, please follow the steps in alphabetical order.
"""

st.markdown("""
##### "If you know the enemy and know yourself; in a hundred battles, you will never be defeated" - Sun Tzu.
""")
#st.markdown(f"<p class='highlight'>{sentence1}</p>", unsafe_allow_html=True)

st.markdown(f"<p class='highlight'>{sentence2}</p>", unsafe_allow_html=True)
st.markdown(f"<p class='highlight'>{sentence3}</p>", unsafe_allow_html=True)

st.markdown("""
### How to use this Tool:
1. Select a process from the sidebar (listed in order that they should be followed).
2. Complete each step as guided.
3. Download reports and results as needed.
""")

st.info("To begin, choose a process from the sidebar on the left.")
