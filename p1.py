import tkinter as tk
from tkinter import *
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score
from PIL import Image, ImageTk

# Load dataset
lung_data = pd.read_csv("survey lung cancer.csv")
lung_data.GENDER = lung_data.GENDER.map({"M": 1, "F": 2})
lung_data.LUNG_CANCER = lung_data.LUNG_CANCER.map({"YES": 1, "NO": 2})

# Splitting the dataset
x = lung_data.iloc[:, 0:-1]
y = lung_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(C=10, kernel='rbf', probability=True, random_state=9),
    "Naive Bayes": GaussianNB()
}

selected_model = None  # Placeholder for the selected model


# Function to train the selected model
def train_model():
    global selected_model
    # Get selected model from dropdown
    selected_model_name = model_var.get()
    selected_model = models[selected_model_name]

    # Train the model
    selected_model.fit(x_train, y_train)

    # Calculate accuracy
    y_pred = selected_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # Display accuracy
    result_text.set(f"Model Trained: {selected_model_name}\nAccuracy: {acc * 100:.2f}%")


# Function to predict cancer using the trained model
def predict_cancer():
    global selected_model

    if not selected_model:
        result_text.set("Please train a model first!")
        return

    # Get inputs
    try:
        inputs = [
            1 if gender_var.get() == "M" else 2,  # GENDER
            int(age_var.get()),  # AGE
            int(smoking_var.get()),  # SMOKING
            int(yellow_fingers_var.get()),  # YELLOW_FINGERS
            int(anxiety_var.get()),  # ANXIETY
            int(peer_pressure_var.get()),  # PEER_PRESSURE
            int(chronic_disease_var.get()),  # CHRONIC DISEASE
            int(fatigue_var.get()),  # FATIGUE
            int(allergy_var.get()),  # ALLERGY
            int(wheezing_var.get()),  # WHEEZING
            int(alcohol_var.get()),  # ALCOHOL CONSUMING
            int(coughing_var.get()),  # COUGHING
            int(shortness_breath_var.get()),  # SHORTNESS OF BREATH
            int(swallowing_var.get()),  # SWALLOWING DIFFICULTY
            int(chest_pain_var.get()),  # CHEST PAIN
        ]

        # Convert inputs to a DataFrame for prediction
        input_data = pd.DataFrame([inputs], columns=x.columns)

        # Make a prediction
        prediction = selected_model.predict(input_data)
        result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer"

        result_text.set(result)
    except ValueError:
        result_text.set("Please enter valid inputs!")


# Create GUI
app = tk.Tk()
app.title("Lung Cancer Prediction Models")
app.geometry("1200x750")  # Adjusted window size
app.resizable(False, False)

# Load and set background image
img = Image.open(r"C:\Users\PRANAV KANSAL\Desktop\Lung-Cancer-Detection-main\Lung-Cancer-Detection-main\bg_image.jpg")

# Resize image to fit the right side of the window
original_width, original_height = img.size
right_width = 600  # Width of the right section
window_height = 750  # Total height of the window

# Calculate new dimensions while preserving aspect ratio
aspect_ratio = original_width / original_height
if aspect_ratio > 1:
    # Wider image, scale by height
    new_height = window_height
    new_width = int(aspect_ratio * new_height)
else:
    # Taller image, scale by width
    new_width = right_width
    new_height = int(new_width / aspect_ratio)

# Resize the image
img = img.resize((right_width, window_height), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img)

# Place the image on the right side
bg_label = tk.Label(app, image=photo)
bg_label.place(x=600, y=0, width=right_width, height=window_height)

# Frame for inputs
frame = tk.Frame(app, bg="black", padx=20, pady=20)
frame.place(x=20, y=0, width=580, height=750)

# Input variables
gender_var = tk.StringVar()
age_var = tk.StringVar()
smoking_var = tk.StringVar()
yellow_fingers_var = tk.StringVar()
anxiety_var = tk.StringVar()
peer_pressure_var = tk.StringVar()
chronic_disease_var = tk.StringVar()
fatigue_var = tk.StringVar()
allergy_var = tk.StringVar()
wheezing_var = tk.StringVar()
alcohol_var = tk.StringVar()
coughing_var = tk.StringVar()
shortness_breath_var = tk.StringVar()
swallowing_var = tk.StringVar()
chest_pain_var = tk.StringVar()

# Dropdown menu for model selection
tk.Label(frame, text="Select Model", font=("Arial", 12), bg="black", fg="white").grid(row=0, column=0, sticky="w", pady=5)
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(frame, textvariable=model_var, values=list(models.keys()), state="readonly", width=25)
model_dropdown.grid(row=0, column=1, pady=5)

# Train button
train_button = tk.Button(frame, text="Train Model", command=train_model, font=("Arial", 12), bg="lightblue", fg="black", width=20)
train_button.grid(row=1, column=0, columnspan=2, pady=10)

# Function to create input fields
def create_input_field(row, label_text, variable, is_gender=False):
    label = tk.Label(frame, text=label_text, font=("Arial", 12), bg="black", fg="white")
    label.grid(row=row, column=0, sticky="w", pady=5)
    if is_gender:
        entry = ttk.Combobox(frame, textvariable=variable, values=["M", "F"], state="readonly", width=10)
    else:
        entry = ttk.Entry(frame, textvariable=variable, width=10)
    entry.grid(row=row, column=1, pady=5)

create_input_field(2, "GENDER (M/F)", gender_var, is_gender=True)
create_input_field(3, "AGE (Years)", age_var)
create_input_field(4, "SMOKING (1=No, 2=Yes)", smoking_var)
create_input_field(5, "YELLOW FINGERS (1=No, 2=Yes)", yellow_fingers_var)
create_input_field(6, "ANXIETY (1=No, 2=Yes)", anxiety_var)
create_input_field(7, "PEER PRESSURE (1=No, 2=Yes)", peer_pressure_var)
create_input_field(8, "CHRONIC DISEASE (1=No, 2=Yes)", chronic_disease_var)
create_input_field(9, "FATIGUE (1=No, 2=Yes)", fatigue_var)
create_input_field(10, "ALLERGY (1=No, 2=Yes)", allergy_var)
create_input_field(11, "WHEEZING (1=No, 2=Yes)", wheezing_var)
create_input_field(12, "ALCOHOL CONSUMING (1=No, 2=Yes)", alcohol_var)
create_input_field(13, "COUGHING (1=No, 2=Yes)", coughing_var)
create_input_field(14, "SHORTNESS OF BREATH (1=No, 2=Yes)", shortness_breath_var)
create_input_field(15, "SWALLOWING DIFFICULTY (1=No, 2=Yes)", swallowing_var)
create_input_field(16, "CHEST PAIN (1=No, 2=Yes)", chest_pain_var)

# Predict button
predict_button = tk.Button(frame, text="Predict Lung Cancer", command=predict_cancer, font=("Arial", 12), bg="lightblue", fg="black", width=20)
predict_button.grid(row=17, column=0, columnspan=2, pady=20)

# Display result
result_text = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_text, font=("Arial", 14, "bold"), bg="black", fg="white", justify="left")
result_label.grid(row=18, column=0, columnspan=2, pady=10)

# Run the GUI
app.mainloop()
