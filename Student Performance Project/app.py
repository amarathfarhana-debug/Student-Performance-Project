
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
        #D2C2F4 0%,
        #C6B4EE 50%,
        #D8C9F5 100%
    );
    color: #252344;
}


/* Main headings */
h1, h2, h3 {
    color: #35256F !important;
    font-weight: 700 !important;
}

/* Normal text */
p, label, .stMarkdown, .stCaption {
    color: #35334D;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #C9B8F0 0%,
        #D3C6F3 100%
    );
    border-right: 1px solid #B9A5E8;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #35256F !important;
}

/* Buttons */

/* Button styling */
.stButton > button,
.stDownloadButton > button {
    background: #9B88DD;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

/* Hover colour */
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #8873CF;
    color: #FFFFFF !important;
    border: 1px solid #8873CF;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #FFFFFF;
    color: #252344;
    border-radius: 8px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0D9F5;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 3px 10px rgba(53, 37, 111, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #5B527A !important;
}

[data-testid="stMetricValue"] {
    color: #35256F !important;
    font-weight: 700;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E0D9F5;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #DDD5F4;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}


/* Sidebar button styling */
[data-testid="stSidebar"] .stButton > button {
    background: #8975D5;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 16px;
    transition: background 0.2s ease;
}

/* Hover colour */
[data-testid="stSidebar"] .stButton > button:hover {
    background: #7058C4;
    color: #FFFFFF !important;
    border: none;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(120deg, #B9A7EF, #A7D9D5);
            padding: 26px; border-radius: 18px; margin-bottom: 22px;">

<h2 style="background: linear-gradient(90deg, #49328F, #247F83);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           font-size: 29px; font-weight: 800; margin-bottom: 10px;">
🎓 Student Academic Performance
</h2>

<p style="color: #37305F; font-size: 15px; font-weight: 500;
          margin-bottom: 0; line-height: 1.7;">
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

    st.success("Prediction Completed Successfully!")

    
    
    import plotly.express as px

    st.subheader("📊 Student Performance Analysis")

    chart_data = pd.DataFrame({
        "Category": [
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Assignment Score"
        ],
        "Score": [
            hours / 24 * 100,
            att,
            previous,
            assignment
        ]
    })

    fig = px.bar(
        chart_data,
        x="Category",
        y="Score",
        color="Category",
        color_discrete_sequence=[
    "#A78BFA",  # Soft Purple
    "#67D5E5",  # Soft Cyan
    "#F6C66B",  # Soft Orange
    "#F58BA7"   # Soft Pink
],
        range_y=[0, 100],
        text="Score",
        title="Student Academic Performance"
    )

    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
    fig.update_traces(width=0.5)
    
    fig.update_layout(
    showlegend=True,
    legend_title_text="Categories",
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02
    ),
    yaxis_title="Score (%)",
    xaxis_title="Performance Factors"
)

    st.plotly_chart(fig, use_container_width=False, width=650)


   
    st.subheader("📚 Personalized Improvement Plan")

    if rounded_marks >= 75:
        st.success("🌟 Performance: Excellent!")
        st.markdown("""
        **How to improve further:**
        - Practise advanced questions and mock tests.
        - Revise important topics regularly.
        - Maintain consistent study habits.
        - Submit assignments on time.
        """)

    elif rounded_marks >= 50:
        st.info("👍 Performance: Good!")
        st.markdown("""
        **Recommended improvement plan:**
        - Increase daily study time gradually.
        - Revise difficult topics regularly.
        - Improve attendance and class participation.
        - Complete assignments before deadlines.
        - Take weekly self-tests.
        """)

    else:
        st.warning("📚 Performance: Needs Improvement")
        st.markdown("""
        **Recommended improvement plan:**
        - Follow a daily study timetable.
        - Start with basic concepts.
        - Practise questions every day.
        - Ask teachers for help with difficult topics.
        - Attend classes regularly.
        - Review mistakes every week.
        """)

    st.info(
        "Note: Predictions are estimates based on the model's training data."
    )


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
                height=320,
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


