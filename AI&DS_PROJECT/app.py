import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import numpy as np

st.set_page_config(page_title="Personalized Fitness Recommender", layout="wide")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload HAR Dataset File (X_train.csv format)", type=["csv"])

# ----------------------------
# FUNCTIONS
# ----------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def get_workout_plan(steps, calories, bmi=None):
    if steps < 4000:
        workout = "Light Yoga + 20-min Walk"
    elif 4000 <= steps < 8000:
        workout = "30-min Jog + Core Strength (planks, leg raises, seated twists)"
    else:
        workout = "HIIT (jump squats, burpees, high knees) + Strength Training (push-ups, dumbbells, resistance bands)"

    if bmi:
        if bmi > 25:
            workout += " (Focus on Fat Burn)"
        elif bmi < 18.5:
            workout += " (Include Weight Gain Routine)"

    return workout

def get_diet_plan(calories):
    if calories < 200:
        return "Light meals (salads, fruits, low-carb dinner)"
    elif 200 <= calories <= 400:
        return "Balanced diet (protein + carbs + veggies)"
    else:
        return "High-protein meals (eggs, lentils, grilled chicken)"

# ----------------------------
# Activity label mapping
# ----------------------------
activity_names = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# ----------------------------
# MAIN CONTENT
# ----------------------------

st.title("Personalized Fitness Recommendation System")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Upload & Preprocess", "2. Visualize", "3. Random Forest Report", "4. Recommendation", "5. Summary"])

with tab1:
    st.subheader("Upload & Preview")
    if uploaded_file:
        df = load_data(uploaded_file)
        activity_col = "Activity" 
        df[activity_col] = df[activity_col].map(activity_names)
        st.write(df.head())
        st.success("File uploaded successfully!")
    else:
        st.warning("Please upload a file to continue.")

with tab2:
    st.subheader(" Activity Visualizations")
 
    if uploaded_file:
        activity_col = "Activity" 
        
        # --- Bar Chart ---
        st.markdown("Activity Count")
        fig_bar, ax_bar = plt.subplots()
        sns.countplot(x=activity_col, hue=activity_col, data=df, order=df[activity_col].value_counts().index, ax=ax_bar)
        ax_bar.set_xlabel("Activity")
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("Activity Visualizations")
        ax_bar.tick_params(axis='x', rotation=45)
        st.pyplot(fig_bar)


        # --- Pie Chart ---
        st.markdown("Activity Distribution")
        activity_counts = df[activity_col].value_counts()
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title("Activity Proportion")
        ax_pie.axis('equal')
        st.pyplot(fig_pie)


        # --- Box Plot ---
        st.markdown("Box Plot")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected_metric = st.selectbox("Select Metric for Box Plot", numeric_cols)
        st.markdown(f"Box Plot of {selected_metric} by Activity")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=activity_col, y=selected_metric, ax=ax)
        ax.set_title(f"{selected_metric} across Activities")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        

        # --- Scatter Plot ---
        st.markdown("Scatter Plot")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=activity_col, ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis} by Activity")
        st.pyplot(fig)
    else:
       st.warning("Please upload a file to view visualizations.")

with tab3:
    st.subheader("Random Forest Classification Report")

    if uploaded_file:
        activity_col = "Activity" 
        X = df.drop(columns=[activity_col])
        X = X.select_dtypes(include=[np.number])
        y = df[activity_col]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        report_dict = classification_report(y_test, y_pred, target_names=[str(cls) for cls in le.classes_], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()

        st.subheader("Classification Report for Random Forest:")
        st.dataframe(report_df.style.format(precision=2))

        st.markdown("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", xticklabels=le.classes_, yticklabels=le.classes_)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

with tab4:
    st.subheader("Personalized Recommendation")

    steps = st.slider("Steps Today", 1000, 20000, 6000, 500)
    calories = st.slider("Calories Burned", 50, 1000, 300, 50)
    bmi = st.slider("BMI", 10.0, 40.0, 24.0, 0.1)

    workout = get_workout_plan(steps, calories, bmi)
    diet = get_diet_plan(calories)

    st.markdown(f"*Workout Plan:* {workout}")
    st.markdown(f"*Diet Plan:* {diet}")

with tab5:
    st.subheader("Fitness Summary & Report Generator")

    report_type = st.radio("Select Report Type", ["Weekly", "Monthly"])

    if report_type == "Weekly":
        week_data = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Steps': [3000, 4500, 7000, 6500, 8000, 9500, 10000],
            'Calories': [150, 200, 300, 280, 350, 400, 420],
            'Distance (km)': [2.2, 3.5, 5.0, 4.8, 6.0, 7.2, 8.0]
        })

        st.markdown("Weekly Summary Chart")
        st.line_chart(week_data.set_index("Day"))

        total_steps = week_data['Steps'].sum()
        avg_calories = week_data['Calories'].mean()
        active_days = sum(week_data['Steps'] > 6000)
        sedentary_days = 7 - active_days

        summary_text = f"""
Weekly Fitness Report:

Total Steps: {total_steps}
Average Calories Burned: {avg_calories:.2f}
Active Days (>6000 steps): {active_days}
Sedentary Days: {sedentary_days}

Recommendation: {'Increase activity on weekdays' if active_days < 5 else 'Keep up the great work!'}
Nutrition Tip: Stay hydrated and avoid late-night snacks.
"""

    else:
        days = pd.date_range(start='2025-04-01', periods=30)
        steps = np.random.randint(3000, 12000, size=30)
        calories = np.random.randint(100, 600, size=30)
        df_month = pd.DataFrame({'Date': days, 'Steps': steps, 'Calories': calories})

        st.markdown("### Monthly Trend")
        st.line_chart(df_month.set_index("Date"))

        total_steps = df_month['Steps'].sum()
        avg_steps = df_month['Steps'].mean()
        high_activity_days = sum(df_month['Steps'] > 8000)

        summary_text = f"""
Monthly Fitness Report:

Total Steps: {total_steps}
Average Daily Steps: {avg_steps:.2f}
High Activity Days (>8000 steps): {high_activity_days}

Recommendation: {'Great consistency! Maintain streak.' if high_activity_days >= 15 else 'Try to hit 8k+ steps on more days.'}
Tip: Add variety to workouts to avoid plateaus.
"""

    st.text_area("Report Preview", summary_text, height=250)

    st.download_button(
        label="Download Report as TXT",
        data=summary_text,
        file_name=f"fitness_{report_type.lower()}_report.txt"
    )