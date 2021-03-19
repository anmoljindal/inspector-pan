import streamlit as st
from streamlit.script_runner import RerunException

import cv2
import re
from PIL import Image
from streamlit.type_util import is_graphviz_chart

name_regex = '[A-Za-z]{2,25}( [A-Za-z]{2,25})?'
name_regex = re.compile(name_regex)

pan_regex = r'^[A-Za-z]{5}[0-9]{4}[A-Za-z]$'
pan_regex = re.compile(pan_regex)

st.title("PAN Card Reader")
st.markdown("submit and verify your pan card")

pan_card_image = st.file_uploader("Upload image", type=['png', 'jpg'], accept_multiple_files=False)
name = st.text_input("Name")
pan = st.text_input("PAN number")
fathers_name = st.text_input("Father's Name")
dob = st.date_input("Date of Birth")

def read_image(image):
    try:
        image = Image.open(image).convert('RGB')
        return image
    except:
        return

def verify_name(name):
    is_valid = bool(name_regex.fullmatch(name))
    return is_valid

def standardize_name(name):
    name = " ".join(name.lower().split())
    return name

def verify_pan(pan):
    is_valid = bool(pan_regex.fullmatch(pan))
    return is_valid

def verification_failure(message):

    st.error(f"PAN Card Verification failed: {message}")
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

if st.button("verify"):
    st.markdown(f'''
    PAN: {pan}\n
    name: {name}\n
    fathers_name: {fathers_name}\n
    dob: {dob}\n
    image: {pan_card_image}
    ''')
    ## preverification checks

    if (pan_card_image is None) or (len(name.strip())==0) or (len(fathers_name.strip())==0):
        verification_failure("missing details")

    image = read_image(pan_card_image)
    if image is None:
        verification_failure("invalid image")
    
    name = standardize_name(name)
    fathers_name = standardize_name(fathers_name)
    if not verify_name(name) or not verify_name(fathers_name):
        verification_failure("incorrect name")
    
    if not verify_pan(pan):
        verification_failure("incorrect pan number")

    st.image(image)
    
    # file is valid image
    # name, father name, pan card number, dob are correct format

    ## verify vs the input

    ## store in a database