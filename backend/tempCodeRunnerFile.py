from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import os
from model import find_best_donor_with_decision_tree
from pdf_extractor import extract_pdf_data  # Import the extraction function

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
           static_folder='../css file' )

# Folder to store uploaded PDFs
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'patient-forms')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the donor data
donor_df = pd.read_csv('backend/donor list.csv')

# Render the main form page (for manual input)
@app.route('/')
def project_page():
    # Render the index1.html from the project_page folder
    return render_template('index1.html')
@app.route('/templates/index.html')
def index():
    return render_template('index.html')

# Route for manual form input (unchanged)
@app.route('/predict', methods=['POST'])
def predict():
    tissue_type = request.form['tissue_type']
    medical_history = request.form['medical_history']
    demographics = request.form['demographics']
    age = int(request.form['age'])
    sex = request.form['sex']
    weight = int(request.form['weight'])
    height = int(request.form['height'])
    waiting_time = int(request.form['waiting_time'])
    urgency = request.form['urgency']

    # Match donor using manual input
    best_donor = find_best_donor_with_decision_tree(donor_df, tissue_type, medical_history, demographics, age, sex, weight, height, waiting_time, urgency)

    if best_donor is not None:
        return render_template('result.html', donor_data=best_donor.to_dict())
    else:
        return render_template('result.html', error="No suitable donor found")

# Route for PDF upload
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract patient data from the uploaded PDF
        patient_data = extract_pdf_data(filepath)

        # Log extracted patient data for debugging
        print("Extracted Patient Data:", patient_data)

        # Match donor using the extracted PDF data
        best_donor = find_best_donor_with_decision_tree(donor_df, **patient_data)

        if best_donor is not None:
            return render_template('result.html', donor_data=best_donor.to_dict())
        else:
            return render_template('result.html', error="No suitable donor found")

if __name__ == '__main__':
    app.run(debug=True)
