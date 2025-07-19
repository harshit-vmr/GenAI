import requests
import json
import streamlit as st


def get_groq_response(language, input_text):
  json_body={
    "input": {"language": language,"text": input_text},
    "config": {},
    "kwargs": {}
  }
  json_data = json.dumps(json_body)
  response=requests.post("http://127.0.0.1:8000/chain/invoke",json_data)

  print(response.json())
  return response.json()

## Streamlit app
st.title("LLM Application Using LCEL")
language = st.selectbox("Select Language", ["French", "Spanish", "German", "Hindi", "Russian"])
input_text=st.text_input("Enter the text you want to convert to french")

if st.button("Translate"):
  st.write(get_groq_response(language, input_text))
