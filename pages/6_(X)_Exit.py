import streamlit as st

#st.header("❌ Exit")


# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size:30px !important;
        color:#FF5733;
        #color:#2E86C1;     
        text-align:center;
    }
    .subtitle {
        font-size:22px !important;
        #color:#FF5733;
        color:#2E86C1;
        text-align:left;
        
    }
    .highlight {
        font-size:22px !important;
        color:#2E86C1;  
        
    }
    </style>
""", unsafe_allow_html=True)

sentence1 = """
The ultimate goal of this PoC is to help you appreciate the versatality of our method to support cyber risk decision-making and overall risk management. 
There is more to this tool than is shown here.  
"""

st.markdown("<p class='title'>End of Analysis</p>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{sentence1}</p>", unsafe_allow_html=True)

st.markdown("""
#### For more information and further conversation, please use the details below:
1. Vist our webpage: http://crsmethods.com
2. Email 📧 : info@crsmethods.com
3. Phone ☎️ : +44 7577647884
""")

if st.button("Exit App"):
    st.write("The tool has stopped. All your information has been removed from the session.")
    st.stop()
else:
    st.write("Click the Exit button above to end your session.")
