import re
from datetime import date

import cv2
import streamlit as st
from streamlit.script_runner import RerunException

import dbutils
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

placeholder = st.empty()

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

    placeholder.error(f"PAN Card Verification failed: {message}")

def verification_success():

    placeholder.success(f"verification successful")

def add_to_db(pan, name, fathers_name, dob):
    product_sql = f'INSERT INTO pan (id, name, fathers_name, dob) VALUES ("{pan}", "{name}", "{fathers_name}", "{dob}")'
    dbutils.run_query(product_sql)

def verify_details(text_blobs, pan, name, fathers_name, dob):

    verified = {
        "name":False,
        "fathers_name":False,
        "pan":False,
        "dob":False
    }

    for text_blob in text_blobs:
        if name == text_blob:
            verified['name'] = True
        if fathers_name == text_blob:
            verified['fathers_name'] = True
        if pan in text_blob:
            verified['pan'] = True
        if dob == text_blob:
            verified['dob'] = True

    for k,v in verified.items():
        if v is False:
            verification_failure(f"{k} did not match with image")
            return False

    verification_success()
    add_to_db(pan, name, fathers_name, dob)
    return True

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
    pan = pan.lower()

    dob = standardize_dob(dob)

    #verify and match text with image
    text_blobs = image_process.read_text_process(image)
    verification_flag = verify_details(text_blobs, pan, name, fathers_name, dob)