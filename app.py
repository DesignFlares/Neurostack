from flask import Flask, render_template, request, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = "super_secret_key"


# ==============================
# Load New Root Disease Model
# ==============================
disease_model = joblib.load("models/disease_classifier_model.pkl")
disease_label_encoder = joblib.load("models/disease_label_encoder.pkl")
disease_feature_columns = joblib.load("models/disease_features.pkl")


# ==============================
# Load Migraine Model
# ==============================
migraine_model = joblib.load("models/mlp_model.pkl")
migraine_scaler = joblib.load("models/scaler_mlp.pkl")

migraine_feature_columns =  [
"Age","Duration","Frequency","Location","Character","Intensity",
"Nausea","Vomit","Phonophobia","Photophobia","Visual","Sensory",
"Dysphasia","Dysarthria","Vertigo","Tinnitus","Hypoacusis",
"Diplopia","Defect","Conscience","Paresthesia","DPF"
]

# ==============================
# Load Epilepsy Model
# ==============================
epilepsy_model = joblib.load("models/reduced_logistic_model.pkl")

epilepsy_feature_columns = [
    "age",
    "seizure_frequency",
    "seizure_duration",
    "eeg_abnormality_detected",
    "family_history_of_epilepsy",
    "loss_of_consciousness",
    "muscle_stiffness"
]

# ==============================
# Load Meningitis Model
# ==============================
meningitis_model = joblib.load("models/meningitis_risk_model.pkl")
meningitis_scaler = joblib.load("models/meningitis_risk_scaler.pkl")

meningitis_feature_columns = [
    "Age",
    "Gender",
    "WBC_Count",
    "Protein_Level",
    "Glucose_Level",
    "Pathogen_Present",
    "Diagnosis",
    "Hemoglobin",
    "WBC_Blood_Count",
    "Platelets",
    "CRP_Level"
]

# ==============================
# Load Multiple Sclerosis Model
# ==============================
ms_model = joblib.load("models/ms_prediction_model.pkl")

ms_feature_columns = [
    "Gender",
    "Age",
    "Schooling",
    "Varicella",
    "Initial_Symptom",
    "Mono_or_Polysymptomatic",
    "Oligoclonal_Bands",
    "ULSSEP",
    "VEP",
    "Periventricular_MRI",
    "Cortical_MRI",
    "Infratentorial_MRI",
    "Spinal_Cord_MRI",
    "Initial_EDSS"
]




# ==============================
# Home
# ==============================
@app.route("/")
def home():
    return render_template("index.html", features=disease_feature_columns)


# ==============================
# Disease Prediction
# ==============================
@app.route("/predict", methods=["POST"])
def predict_disease():
    try:

        input_data = []
        root_inputs_dict = {}

        for feature in disease_feature_columns:

            value = request.form.get(feature)

            if value is None or value == "":
                value_float = 0
            else:
                value_float = float(value)

            input_data.append(value_float)

            root_inputs_dict[feature.lower()] = value_float

        # ==============================
        # Healthy Case
        # ==============================
        if sum(input_data) == 0:
            return render_template(
                "result.html",
                prediction="Healthy Person",
                confidence=100
            )

        input_array = np.array([input_data])

        prediction = disease_model.predict(input_array)
        probabilities = disease_model.predict_proba(input_array)

        disease_name = disease_label_encoder.inverse_transform(prediction)[0]
        confidence = round(float(max(probabilities[0])) * 100, 2)

        session["root_inputs"] = root_inputs_dict

        # ==============================
        # MIGRAINE CASE
        # ==============================
        if disease_name.lower() == "migraine":
            return render_template(
                "migraine_extra.html",
                migraine_features=migraine_feature_columns,
                confidence=confidence
            )

        # ==============================
        # EPILEPSY CASE
        # ==============================
        if disease_name.lower() == "epilepsy":
            return render_template(
                "epilepsy_extra.html",
                epilepsy_features=epilepsy_feature_columns
            )

        # ==============================
        # MENINGITIS CASE
        # ==============================
        if disease_name.lower() == "meningitis":
            return render_template(
                "meningitis_extra.html",
                meningitis_features=meningitis_feature_columns,
                confidence=confidence
            )

        # ==============================
        # MULTIPLE SCLEROSIS CASE
        # ==============================
        if disease_name.lower() == "multiple sclerosis":
            return render_template(
                "ms_extra.html",
                ms_features=ms_feature_columns
            )

        # ==============================
        # OTHER CASE
        # ==============================
        return render_template(
            "result.html",
            prediction=disease_name,
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ==============================
# Migraine Type Prediction
# ==============================


@app.route("/predict_migraine_type", methods=["POST"])
def predict_migraine_type():
    try:
        values = []

        for feature in migraine_feature_columns:
            value = request.form.get(feature)
            values.append(float(value))

        # convert to numpy array
        arr = np.array(values).reshape(1, -1)

        # scale input
        arr_scaled = migraine_scaler.transform(arr)

        # prediction
        prediction = migraine_model.predict(arr_scaled)
        prob = migraine_model.predict_proba(arr_scaled)

        migraine_type = prediction[0]

        # confidence
        confidence = round(float(prob.max()) * 100, 2)

        return render_template(
            "result.html",
            prediction=migraine_type,
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route("/predict_epilepsy", methods=["POST"])
def predict_epilepsy():
    try:
        input_data = []

        for feature in epilepsy_feature_columns:
            value = request.form.get(feature)
            input_data.append(float(value))

        input_array = np.array([input_data])

        prediction = epilepsy_model.predict(input_array)[0]
        prob = epilepsy_model.predict_proba(input_array)[0]
        if prediction==1:
            epilepsy_type="generalized"
        else:
            epilepsy_type="focal"
        confidence = round(float(max(prob)) * 100, 2)

        return render_template(
            "result.html",
            prediction=epilepsy_type,
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ==============================
# Meningitis Risk Prediction
# ==============================
@app.route("/predict_meningitis", methods=["POST"])
def predict_meningitis():
    try:

        input_data = []

        input_data = []

        for feature in meningitis_feature_columns:
            if feature == "WBC_Blood_Count":
        # copy value from WBC_Count
                value = request.form.get("WBC_Count")
            else:
                 value = request.form.get(feature)

            input_data.append(float(value))

        input_array = np.array([input_data])

        # Scale input
        input_scaled = meningitis_scaler.transform(input_array)

        prediction = meningitis_model.predict(input_scaled)[0]
        prob = meningitis_model.predict_proba(input_scaled)[0]

        confidence = round(float(max(prob)) * 100, 2)

        risk_map = {
            1: "Low Risk",
            2: "Moderate Risk",
            3: "High Risk"
        }

        return render_template(
            "result.html",
            prediction=f"Meningitis Risk: {risk_map[prediction]}",
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"
    
# ==============================
# Multiple Sclerosis Prediction
# ==============================
@app.route("/predict_ms", methods=["POST"])
def predict_ms():
    try:

        input_data = []

        for feature in ms_feature_columns:
            value = request.form.get(feature)
            input_data.append(float(value))

        input_array = np.array([input_data])

        prediction = ms_model.predict(input_array)[0]
        prob = ms_model.predict_proba(input_array)[0]

        confidence = round(float(max(prob)) * 100, 2)

        # Convert model output to medical meaning
        if prediction == 0:
            result = "CDMS (Converted to Multiple Sclerosis)"
        else:
            result = "Non-CDMS (No conversion to MS)"

        return render_template(
            "result.html",
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"
# ==============================
# Direct Pages for Navbar
# ==============================

@app.route("/migraine")
def migraine_page():
    return render_template(
        "migraine_extra.html",
        migraine_features=migraine_feature_columns
    )


@app.route("/epilepsy")
def epilepsy_page():
    return render_template(
        "epilepsy_extra.html",
        epilepsy_features=epilepsy_feature_columns
    )


@app.route("/meningitis")
def meningitis_page():
    return render_template(
        "meningitis_extra.html",
        meningitis_features=meningitis_feature_columns
    )


@app.route("/multiple_sclerosis")
def ms_page():
    return render_template(
        "ms_extra.html",
        ms_features=ms_feature_columns
    )

if __name__ == "__main__":
    app.run(debug=True)