import re
from datetime import date

import cv2
import streamlit as st
from streamlit.script_runner import RerunException

import image_process

name_regex = '[A-Za-z]{2,25}( [A-Za-z]{2,25})?'
name_regex = re.compile(name_regex)

pan_regex = r'^[A-Za-z]{5}[0-9]{4}[A-Za-z]$'
pan_regex = re.compile(pan_regex)

st.title("PAN Card Reader")
st.markdown("submit and verify your pan card")

pan_card_image = st.file_uploader("Upload image", type=['png', 'jpg'], accept_multiple_files=False)

cols = st.beta_columns(2)
with cols[0]:
    pan = st.text_input("PAN number")
with cols[1]:
    min_value = date(1921, 1, 1)
    dob = st.date_input("Date of Birth", min_value=min_value)
name = st.text_input("Name")
fathers_name = st.text_input("Father's Name")

def standardize_name(name: str) -> str:
    name = " ".join(name.lower().split())
    return name

def standardize_dob(dob: date) -> str:
    dob = dob.strftime("%d/%m/%Y")
    return dob

def verify_name(name: str) -> bool:
    is_valid = bool(name_regex.fullmatch(name))
    return is_valid

def verify_pan(pan: str) -> bool:
    is_valid = bool(pan_regex.fullmatch(pan))
    return is_valid

def verification_failure(message: str):

    st.error(f"PAN Card Verification failed: {message}")

def verification_success():

    st.success(f"verification successful")

def add_to_db(pan, name, fathers_name, dob):
    product_sql = "INSERT INTO pan (id, name, fathers_name, dob) VALUES (?, ?, ?, ?)"
    cur.execute(product_sql, (pan, name, fathers_name, dob))

failure_status = False

if st.button("verify"):
    ## preverification checks

    if (pan_card_image is None) or (len(name.strip())==0) or (len(fathers_name.strip())==0):
        verification_failure("missing details")
        failure_status = True

    image = image_process.read_image(pan_card_image)
    if image is None:
        verification_failure("invalid image")
        failure_status = True
    
    name = standardize_name(name)
    fathers_name = standardize_name(fathers_name)
    if not verify_name(name) or not verify_name(fathers_name):
        verification_failure("incorrect name")
        failure_status = True
    
    if not verify_pan(pan):
        verification_failure("incorrect pan number")
        failure_status = True

    dob = standardize_dob(dob)
    image = image_process.fix_orientation(image)
    
    st.image(image) #display the image

    #verify and match text with image
    text_blobs = image_process.read_text_process(image)

    verified = {
        "name":False,
        "fathers_name":False,
        "pan":False,
        "dob":False
    }

    for text_blob in text_blobs:
        if name in text_blob:
            verified['name'] = True
        if fathers_name in text_blob:
            verified['fathers_name'] = True
        if pan in text_blob:
            verified['pan'] = True
        if dob in text_blob:
            verified['dob'] = True

    for k,v in verified.items():
        if v is False:
            verification_failure(f"{k} did not match with image")
            failure_status = True

    if not failure_status:
        verification_success()
        add_to_db(pan, name, fathers_name, dob)