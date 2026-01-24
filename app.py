import streamlit as st
from main import answer_query

st.set_page_config(
    page_title="Cotton Mix Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Page body */
        body {
            background-color: #121212;
            color: #FFFFFF;
        }

        /* Submit button */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            height: 3em;
            width: 10em;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }

        /* Input box */
        .stTextInput>div>input {
            height: 2em;
            font-size: 16px;
        }

        /* Result box */
        .result-box {
            background-color: #1E1E1E;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            margin-top: 10px;
            font-size: 16px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* Headers */
        .header {
            color: #FFD700;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .subheader {
            color: #FFA500;
            font-size: 20px;
            font-weight: bold;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Cotton Mix Quality Assistant 🌿</div>', unsafe_allow_html=True)
st.markdown("Ask any question about the cotton dataset, including: quality stats, lot analysis, recommendations, or quality insights.")

st.sidebar.header("Instructions")
st.sidebar.write("""
- Type your question in the input box and click **Submit**.
- The bot will process your query and provide an answer based on the cotton mix dataset.
""")

query = st.text_input("Type your question here:")

if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            answer = answer_query(query)
        st.markdown('<div class="result-box">{}</div>'.format(answer.replace('\n', '<br>')), unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")
