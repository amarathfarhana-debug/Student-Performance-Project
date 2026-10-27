
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Student Academic Performance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #F0F9FF 0%,
        #E0F2FE 45%,
        #CCFBF1 100%
    );
    color: #172554;
}
/* Main title */
h1 {
    color: #172554 !important;
    font-weight: 800 !important;
}

/* Section headings */
h2, h3 {
    color: #0F766E !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    padding-bottom: 6px;
    border-bottom: 2px solid #B8D8D6;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF, #DBEAFE, #CCFBF1);
    border: 1px solid #93C5FD;
    border-left: 5px solid #2563EB;
    padding: 22px 18px;
    border-radius: 16px;
    box-shadow: 0 5px 15px rgba(37, 99, 235, 0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #172554;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #0F766E;
    font-size: 1.8rem;
    font-weight: 800;
}


/* Metric values */
[data-testid="stMetricValue"] {
    color: #0F766E;
    font-weight: 800;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E0F2F1, #F4F7FB);
    border-right: 2px solid #B8D8D6;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172554 !important;
    font-weight: 800;
}

[data-testid="stSidebar"] label {
    color: #172554 !important;
    font-weight: 600;
}

[data-testid="stSidebar"] input {
    border-radius: 8px;
    border: 1px solid #99C7C4;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #0F766E, #115E59);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 12px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(15, 118, 110, 0.20);
    width: 100%;
}

/* Button hover */
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #115E59;
    color: white;
    border: 1px solid #115E59;
}

[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #FFFFFF, #E7F3F3);
    border: 2px dashed #0F766E;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    border-radius: 10px;
}

[data-testid="stFileUploader"] small {
    color: #172554;
}
/* Data table borders */
[data-testid="stDataFrame"] {
    border: 1px solid #D8E3EE;
    border-radius: 10px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #B8D8D6;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(23, 37, 84, 0.08);
}

[data-testid="stDataFrame"] div[role="grid"] {
    border-radius: 12px;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

hr {
    border-color: #B8D8D6;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

.stCaption {
    color: #64748B;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
    background: linear-gradient(135deg, #172554, #0F766E);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
">
    <h2 style="color: white !important; border: none;">
        🎓 Student Academic Performance
    </h2>
    <p style="color: #E0F2FE; margin-bottom: 0;">
        Analyze student data, predict marks, and explore performance insights.
    </p>
</div>
""", unsafe_allow_html=True)

st.title("🎓 Student Academic Performance System")
st.caption("An intelligent dashboard for student performance analysis and prediction.")
st.divider()
st.write("Predict student marks, grade and pass/fail status.")

# Create sample data for a demonstration model
np.random.seed(42)
n = 300

study_hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(40, 100, n)
previous_marks = np.random.uniform(20, 100, n)
assignment_score = np.random.uniform(0, 100, n)

marks = (
    0.25 * previous_marks
    + 2.5 * study_hours
    + 0.15 * attendance
    + 0.20 * assignment_score
    + np.random.normal(0, 5, n)
)
marks = np.clip(marks, 0, 100)

X = pd.DataFrame({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Previous Marks": previous_marks,
    "Assignment Score": assignment_score
})


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, marks, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f} marks")
st.write(f"R² Score: {r2:.2f}")

st.subheader("Student Performance Analysis")

chart_data = pd.DataFrame({
    "Study Hours": X_test["Study Hours"].values,
    "Actual Marks": y_test,
    "Predicted Marks": test_predictions
})

st.write("Study Hours vs Actual and Predicted Marks")
st.line_chart(
    chart_data.set_index("Study Hours")[
        ["Actual Marks", "Predicted Marks"]
    ],
    height=280
)

attendance_chart = pd.DataFrame({
    "Attendance": X_test["Attendance"].values,
    "Predicted Marks": test_predictions
}).sort_values("Attendance")

st.write("Attendance vs Predicted Marks")
st.line_chart(
    attendance_chart.set_index("Attendance"),
    height=280
)

st.sidebar.header("🎓 Student Details")
st.sidebar.caption("Enter student information below.")
st.sidebar.divider()
name = st.sidebar.text_input("Student Name")
hours = st.sidebar.number_input("Daily Study Hours", 0.0, 24.0, 2.0)
att = st.sidebar.number_input("Attendance (%)", 0.0, 100.0, 75.0)
previous = st.sidebar.number_input("Previous Marks", 0.0, 100.0, 60.0)
assignment = st.sidebar.number_input("Assignment Score", 0.0, 100.0, 70.0)

if st.sidebar.button("Predict Performance"):
    student = pd.DataFrame([{
        "Study Hours": hours,
        "Attendance": att,
        "Previous Marks": previous,
        "Assignment Score": assignment
    }])

    predicted_marks = float(np.clip(model.predict(student)[0], 0, 100))
    rounded_marks = round(predicted_marks)

    st.success("Prediction Completed Successfully!")

    st.subheader("📊 Student Performance Result")

    st.metric("Predicted Marks", f"{rounded_marks}/100")

    if rounded_marks >= 75:
        st.success("Performance: Excellent 🌟")
    elif rounded_marks >= 50:
        st.info("Performance: Good 👍")
    else:
        st.warning("Performance: Needs Improvement 📚")

    if rounded_marks >= 90:
        grade = "A+"
    elif rounded_marks >= 80:
        grade = "A"
    elif rounded_marks >= 70:
        grade = "B"
    elif rounded_marks >= 60:
        grade = "C"
    elif rounded_marks >= 50:
        grade = "D"
    else:
        grade = "F"

    result = "Pass" if rounded_marks >= 35 else "Fail"

    st.subheader("🎓 Final Result")

    st.metric("Grade", grade)

    if result == "Pass":
        st.success("Result: PASS ✅")
    else:
        st.error("Result: FAIL ❌")

    import plotly.express as px

    chart_data = pd.DataFrame({
        "Category": [
            "Attendance",
            "Previous Marks",
            "Assignment Score",
            "Predicted Marks"
        ],
        "Score": [
            att,
            previous,
            assignment,
            rounded_marks
        ]
    })

    fig = px.bar(
        chart_data,
        x="Category",
        y="Score",
        title="Student Academic Performance",
        color="Category",
        range_y=[0, 100]
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🎯 Prediction Results")
    st.write("👤 Student Name:", name if name.strip() else "Not provided")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("📊 Predicted Marks", f"{rounded_marks}/100")

    with result_col2:
        grade_color = (
            "#15803D" if grade in ["A+", "A"]
            else "#D97706" if grade in ["B", "C", "D"]
            else "#DC2626"
        )

        st.markdown(
            f"""
            <div style="
                background: white;
                border: 1px solid #B8D8D6;
                border-left: 5px solid {grade_color};
                padding: 18px;
                border-radius: 14px;
                text-align: center;
            ">
                <p style="color: #172554; font-weight: 600;">
                    🎓 Predicted Grade
                </p>
                <h2 style="color: {grade_color}; border: none;">
                    {grade}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )


    with result_col3:
        st.metric("✅ Predicted Result", result)

    st.info(
        "This is a demonstration model trained on generated sample data. "
        "Its predictions are not validated for real students."
    )

st.subheader("📂 Upload Student Data")

uploaded_file = st.file_uploader(
    "Upload a student CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    required_columns = [
        "Student Name",
        "Study Hours",
        "Attendance",
        "Previous Marks",
        "Assignment Score"
    ]

    if all(col in data.columns for col in required_columns):
        st.write("Uploaded Student Data")
        st.dataframe(data)

        features = data[
            [
                "Study Hours",
                "Attendance",
                "Previous Marks",
                "Assignment Score"
            ]
        ]

        predictions = model.predict(features)
        data["Predicted Marks"] = np.round(
            np.clip(predictions, 0, 100)
        )

        def get_grade(mark):
            if mark >= 90:
                return "A+"
            elif mark >= 80:
                return "A"
            elif mark >= 70:
                return "B"
            elif mark >= 60:
                return "C"
            elif mark >= 50:
                return "D"
            else:
                return "F"

        data["Grade"] = data["Predicted Marks"].apply(get_grade)
        data["Result"] = data["Predicted Marks"].apply(
            lambda mark: "Pass" if mark >= 35 else "Fail"
        )
        
        st.markdown("## 📈 Student Performance Dashboard")
        st.caption("Overview of uploaded students' predicted academic performance.")

        total_students = len(data)
        pass_count = (data["Result"] == "Pass").sum()
        fail_count = (data["Result"] == "Fail").sum()
        average_marks = data["Predicted Marks"].mean()

        
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("👨‍🎓 Total Students", total_students)
        col2.metric("📊 Average Marks", f"{average_marks:.1f}/100")
        col3.metric("✅ Pass Students", int(pass_count))
        col4.metric("❌ Fail Students", int(fail_count))

        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("🎓 Grade Distribution")
            grade_counts = (
                data["Grade"]
                .value_counts()
                .reindex(["A+", "A", "B", "C", "D", "F"], fill_value=0)
            )
            st.bar_chart(
                grade_counts,
                height=280,
                color="#0F766E"
            )

        
        with chart2:
            st.subheader("📊 Pass / Fail Distribution")

            result_counts = (
                data["Result"]
                .value_counts()
                .reindex(["Pass", "Fail"], fill_value=0)
            )

            import plotly.express as px

            result_chart = px.bar(
                x=result_counts.index,
                y=result_counts.values,
                color=result_counts.index,
                color_discrete_map={
                    "Pass": "#16A34A",
                    "Fail": "#DC2626"
                },
                labels={
                    "x": "Result",
                    "y": "Number of Students",
                    "color": "Result"
                },
                title="Pass vs Fail Students"
            )

            result_chart.update_layout(
                height=320,
                showlegend=False
            )

            st.plotly_chart(result_chart, use_container_width=True)


        st.subheader("📊 Student Prediction Results")

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        csv_data = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Student Performance Report",
            data=csv_data,
            file_name="Student_Performance_Report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error(
            "CSV must contain: Student Name, Study Hours, "
            "Attendance, Previous Marks, Assignment Score"
        )

    st.divider()

    st.caption(
    "🎓 Student Academic Performance Prediction System | "
    "Machine Learning Project"
       )

    st.caption(
    "Note: Predictions are for demonstration purposes and "
    "should not be treated as verified academic outcomes."
      )
