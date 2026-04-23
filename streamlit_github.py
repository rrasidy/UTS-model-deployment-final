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
    df = pd.DataFrame([kwargs])
    df["experience_score"] = df["internships_completed"] + df["projects_completed"]
    return df


if page == "Placement Prediction":
    st.title("Placement Prediction")

    with st.form("placement_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            branch = st.selectbox("Branch", ["ECE", "IT", "CSE", "CE", "ME"])
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
            tenth_percentage = st.number_input("Tenth Percentage", min_value=0.0, max_value=100.0, step=0.1)
            twelfth_percentage = st.number_input("Twelfth Percentage", min_value=0.0, max_value=100.0, step=0.1)
            backlogs = st.number_input("Backlogs", min_value=0, step=1)
            certifications_count = st.number_input("Certifications Count", min_value=0, max_value=100, step=1)
            hackathons_participated = st.number_input("Hackathons Participated", min_value=0, max_value=100, step=1)

        with col2:
            study_hours_per_day = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, step=0.1)
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1)
            projects_completed = st.number_input("Projects Completed", min_value=0, max_value=100, step=1)
            internships_completed = st.number_input("Internships Completed", min_value=0, max_value=100, step=1)
            coding_skill_rating = st.number_input("Coding Skill Rating", min_value=0, max_value=100, step=1)
            communication_skill_rating = st.number_input("Communication Skill Rating", min_value=0, max_value=100, step=1)
            aptitude_skill_rating = st.number_input("Aptitude Skill Rating", min_value=0, max_value=100, step=1)

        with col3:
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1)
            stress_level = st.number_input("Stress Level", min_value=0, max_value=100, step=1)
            part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
            family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            extracurricular_involvement = st.selectbox("Extracurricular Involvement", ["Unknown", "Low", "Medium", "High"])

        submitted = st.form_submit_button("Predict Placement")

    if submitted:
        input_df = build_input_df(
            gender=gender,
            branch=branch,
            cgpa=cgpa,
            tenth_percentage=tenth_percentage,
            twelfth_percentage=twelfth_percentage,
            backlogs=backlogs,
            study_hours_per_day=study_hours_per_day,
            attendance_percentage=attendance_percentage,
            projects_completed=projects_completed,
            internships_completed=internships_completed,
            coding_skill_rating=coding_skill_rating,
            communication_skill_rating=communication_skill_rating,
            aptitude_skill_rating=aptitude_skill_rating,
            hackathons_participated=hackathons_participated,
            certifications_count=certifications_count,
            sleep_hours=sleep_hours,
            stress_level=stress_level,
            part_time_job=part_time_job,
            family_income_level=family_income_level,
            city_tier=city_tier,
            internet_access=internet_access,
            extracurricular_involvement=extracurricular_involvement
        )

        pred = placement_model.predict(input_df)[0]
        st.success(f"Prediction: {pred}")

        if hasattr(placement_model, "predict_proba"):
            proba = placement_model.predict_proba(input_df)[0]
            st.write("Probability:", proba.tolist())

        fig, ax = plt.subplots()
        ax.bar(
            ["Projects", "Internships", "Hackathons", "Certifications"],
            [projects_completed, internships_completed, hackathons_participated, certifications_count]
        )
        ax.set_title("Experience Profile")
        st.pyplot(fig)

elif page == "Salary Prediction":
    st.title("Salary Prediction")

    with st.form("salary_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="sg")
            branch = st.selectbox("Branch", ["ECE", "IT", "CSE", "CE", "ME"], key="sb")
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, key="sc")
            tenth_percentage = st.number_input("Tenth Percentage", min_value=0.0, max_value=100.0, step=0.1, key="s10")
            twelfth_percentage = st.number_input("Twelfth Percentage", min_value=0.0, max_value=100.0, step=0.1, key="s12")
            backlogs = st.number_input("Backlogs", min_value=0, step=1, key="sbl")
            certifications_count = st.number_input("Certifications Count", min_value=0, max_value=100, step=1, key="scc")
            hackathons_participated = st.number_input("Hackathons Participated", min_value=0, max_value=100, step=1, key="sh")

        with col2:
            study_hours_per_day = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, step=0.1, key="ssh")
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1, key="sap")
            projects_completed = st.number_input("Projects Completed", min_value=0, max_value=100, step=1, key="spc")
            internships_completed = st.number_input("Internships Completed", min_value=0, max_value=100, step=1, key="sic")
            coding_skill_rating = st.number_input("Coding Skill Rating", min_value=0, max_value=100, step=1, key="scr")
            communication_skill_rating = st.number_input("Communication Skill Rating", min_value=0, max_value=100, step=1, key="scom")
            aptitude_skill_rating = st.number_input("Aptitude Skill Rating", min_value=0, max_value=100, step=1, key="sar")

        with col3:
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1, key="ssl")
            stress_level = st.number_input("Stress Level", min_value=0, max_value=100, step=1, key="sst")
            part_time_job = st.selectbox("Part-time Job", ["Yes", "No"], key="spj")
            family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"], key="sfi")
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], key="sct")
            internet_access = st.selectbox("Internet Access", ["Yes", "No"], key="sia")
            extracurricular_involvement = st.selectbox("Extracurricular Involvement", ["Unknown", "Low", "Medium", "High"], key="sei")

        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_df = build_input_df(
            gender=gender,
            branch=branch,
            cgpa=cgpa,
            tenth_percentage=tenth_percentage,
            twelfth_percentage=twelfth_percentage,
            backlogs=backlogs,
            study_hours_per_day=study_hours_per_day,
            attendance_percentage=attendance_percentage,
            projects_completed=projects_completed,
            internships_completed=internships_completed,
            coding_skill_rating=coding_skill_rating,
            communication_skill_rating=communication_skill_rating,
            aptitude_skill_rating=aptitude_skill_rating,
            hackathons_participated=hackathons_participated,
            certifications_count=certifications_count,
            sleep_hours=sleep_hours,
            stress_level=stress_level,
            part_time_job=part_time_job,
            family_income_level=family_income_level,
            city_tier=city_tier,
            internet_access=internet_access,
            extracurricular_involvement=extracurricular_involvement
        )

        pred = float(salary_model.predict(input_df)[0])
        pred = max(0.0, round(pred, 2))
        st.success(f"Predicted Salary (LPA): {pred:.2f}")

else:
    st.title("About")
    st.write("Cloud Streamlit app following notebook logic using saved full pipeline models.")