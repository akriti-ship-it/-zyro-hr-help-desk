import streamlit as st

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")

st.title("💼 Zyro Dynamics HR Help Desk")
st.write("Ask HR-related questions about Zyro Dynamics policies.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        try:
            answer = ask_bot(question)
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")