import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

placement_model = joblib.load("best_classification_model.pkl")
salary_model = joblib.load("best_regression_model.pkl")

st.set_page_config(page_title="Student Placement & Salary Prediction", layout="wide")

st.sidebar.title("Menu Navigasi")
st.sidebar.write("Pilih halaman untuk mulai eksplorasi:")

page = st.sidebar.radio(
    "Go to:",
    ["Placement Prediction", "Salary Prediction", "About"],
    index=0
)


def build_input_df(**kwargs):
    return pd.DataFrame([kwargs])


def student_form(prefix=""):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"{prefix}gender")
        ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}ssc")
        hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}hsc")
        degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}degree")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, key=f"{prefix}cgpa")

    with col2:
        entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}entrance")
        technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}tech")
        soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}soft")
        internship_count = st.number_input("Internship Count", min_value=0, step=1, key=f"{prefix}internships")
        live_projects = st.number_input("Live Projects", min_value=0, step=1, key=f"{prefix}projects")

    with col3:
        work_experience_months = st.number_input("Work Experience Months", min_value=0, step=1, key=f"{prefix}work")
        certifications = st.number_input("Certifications", min_value=0, step=1, key=f"{prefix}certs")
        attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1, key=f"{prefix}attendance")
        backlogs = st.number_input("Backlogs", min_value=0, step=1, key=f"{prefix}backlogs")
        extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"], key=f"{prefix}extra")

    return {
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "internship_count": internship_count,
        "live_projects": live_projects,
        "work_experience_months": work_experience_months,
        "certifications": certifications,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular_activities,
    }


if page == "Placement Prediction":
    st.title("Placement Prediction")

    with st.form("placement_form"):
        payload = student_form(prefix="p_")
        submitted = st.form_submit_button("Predict Placement")

    if submitted:
        input_df = build_input_df(**payload)
        pred = placement_model.predict(input_df)[0]

        pred_label = "Placed" if str(pred) in ["1", "Placed"] else "Not Placed"
        st.success(f"Prediction: {pred_label}")

        if hasattr(placement_model, "predict_proba"):
            proba = placement_model.predict_proba(input_df)[0]
            st.write("Probability:", proba.tolist())

        fig, ax = plt.subplots()
        ax.bar(
            ["Internships", "Projects", "Certifications", "Backlogs"],
            [
                payload["internship_count"],
                payload["live_projects"],
                payload["certifications"],
                payload["backlogs"],
            ]
        )
        ax.set_title("Student Profile")
        st.pyplot(fig)

elif page == "Salary Prediction":
    st.title("Salary Prediction")

    with st.form("salary_form"):
        payload = student_form(prefix="s_")
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_df = build_input_df(**payload)
        pred = float(salary_model.predict(input_df)[0])
        pred = max(0.0, round(pred, 2))
        st.success(f"Predicted Salary (LPA): {pred:.2f}")

        fig, ax = plt.subplots()
        ax.bar(
            ["Internships", "Projects", "Certifications", "Experience (Months)"],
            [
                payload["internship_count"],
                payload["live_projects"],
                payload["certifications"],
                payload["work_experience_months"],
            ]
        )
        ax.set_title("Student Profile")
        st.pyplot(fig)

else:
    st.title("About")
    st.write("Hi! I'm Robert Rasidy, a passionate data scientist and machine learning enthusiast. I have a strong background in deep learning and have worked on various projects involving image classification, natural language processing, and predictive modeling. I enjoy exploring new technologies and applying them to solve real-world problems. In this project, I developed a student placement and salary prediction system using advanced machine learning techniques. Feel free to explore the app and let me know if you have any questions or feedback!        ")