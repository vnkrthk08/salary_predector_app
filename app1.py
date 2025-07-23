import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie


model = joblib.load('rf_model1_compressed.pkl')


education_encoding = {"High School": 1, "Bachelor's":0, "Master's": 2, "PhD": 3}
location_encoding = {"Rural": 0, "Suburban": 1, "Urban": 2}
job_title_encoding = {
    'Clerk':0, 'Technician':1, 'Customer Support':9, 'Data Analyst':5,
    'Software Engineer':2, 'HR Manager':8,
    'Data Scientist':3, 'Product Manager':7, 'Director':4
}
gender_encoding = {'Male': 1, 'Female': 0}


st.set_page_config(page_title="Salary Predictor 💼", page_icon="💸", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .main {
        background: linear-gradient(145deg, #e0e0e0, #ffffff);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .css-1d391kg { background-color: #fff0;}
    a {
        text-decoration: none; /* Removes underline from links */
    }
    </style>
""", unsafe_allow_html=True)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")


st_lottie(lottie_ai, height=200, key="ai")
st.title("💰 Your Financial Compass: AI Salary Insights")
st.markdown("""
    Unlock your earning potential! This **AI-powered tool** helps you discover **realistic salary estimates**
    based on current market trends and your unique profile.
    Simply provide your details below and let our model do the rest!
""")


with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("📚 Education Level", list(education_encoding.keys()))
        experience = st.slider("👔 Years of Experience", 0, 40, 1)
        job_title = st.selectbox("💼 Job Title", list(job_title_encoding.keys()))
    
    with col2:
        location = st.selectbox("📍 Work Location", list(location_encoding.keys()))
        age = st.slider("🎂 Age", 18, 65, 30)
        gender = st.selectbox("👤 Gender", list(gender_encoding.keys()))

    submitted = st.form_submit_button("✨ Get My Salary Estimate!")

    if submitted:

        if age <= experience:
            st.error("❌ Error: Age must be greater than years of experience. Please adjust your inputs.")
            st.stop() # Stop execution if validation fails

        min_starting_age = 18
        if (age - experience) < min_starting_age:
            st.error(f"❌ Error: An individual's age minus their experience should be at least {min_starting_age} (representing a realistic starting age). Please adjust.")
            st.stop()


        input_vector = np.array([[
            education_encoding[education],
            experience,
            location_encoding[location],
            job_title_encoding[job_title],
            age,
            gender_encoding[gender]
        ]])
        input_df = pd.DataFrame(input_vector, columns=['education_level', 'experience', 'location', 'job_title', 'age', 'gender'])

        # Predict
        salary = model.predict(input_df)[0]
        st.success(f"🎉 Great news! Your estimated monthly salary is: **₹{salary:,.2f}**")
        st.info("💡 Keep in mind, this is an estimate based on market data. Actual offers may vary.")


st.markdown("""
<hr>
<div style="text-align:center; padding-top: 20px;">
    <p style="font-size: 1.1em; color: #555;">
        Built with dedication and data by <strong>Aditya Raj</strong> 🚀
    </p>
    <p style="font-size: 0.9em; color: #777;">
        An ECE Undergrad from Birla Institute of Technology, Mesra.
    </p>
    <p style="font-size: 1.0em; margin-top: 15px;">
        <a href="https://www.linkedin.com/in/adityaraj-bit/" target="_blank" style="text-decoration: none; color: #0077B5; margin: 0 10px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/480px-LinkedIn_logo_initials.png" width="20" height="20" style="vertical-align: middle;"> LinkedIn
        </a> |
        <a href="https://www.instagram.com/adityar_a_j_?igsh=MTZicm1qejZmMWg4MQ==/" target="_blank" style="text-decoration: none; color: #C13584; margin: 0 10px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/768px-Instagram_logo_2016.svg.png" width="20" height="20" style="vertical-align: middle;"> Instagram
        </a> |
        <a href="https://aditya-r01.github.io/Portfolio-website/" target="_blank" style="text-decoration: none; color: #337ab7; margin: 0 10px;">
            <img src="https://www.citypng.com/public/uploads/preview/transparent-hd-internet-globe-blue-icon-701751695035228s3pzw5luvt.png?v=2025061313" width="20" height="20" style="vertical-align: middle;"> My Portfolio
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
