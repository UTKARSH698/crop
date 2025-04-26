from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pickle
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder=r'C:\Users\udayd\OneDrive\Desktop\crop 1\Crop-Recommendation-System-Using-Machine-Learning\templates')

try:
    base_path = r"C:\Users\udayd\OneDrive\Desktop\crop 1\Crop-Recommendation-System-Using-Machine-Learning"
    
    model_path = os.path.join(base_path, "model.pkl")
    sc_path = os.path.join(base_path, "standscaler.pkl")
    ms_path = os.path.join(base_path, "minmaxscaler.pkl")
    
    model = pickle.load(open(model_path, 'rb'))
    sc = pickle.load(open(sc_path, 'rb'))
    ms = pickle.load(open(ms_path, 'rb'))
    
    logging.info("Models and scalers loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise

@app.route('/crop-recommendation', methods=['POST'])
def crop_recommendation():
    """
    Provide crop recommendation based on soil parameters.
    """
    try:
        # Validate input data
        data = request.json
        logging.info(f"Received recommendation request: {data}")

        # Validate required fields
        required_fields = ['Nitrogen', 'Phosphorus', 'Potassium', 'Ph', 'Temperature', 'Humidity', 'Rainfall']
        for field in required_fields:
            if field not in data:
                logging.warning(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Prepare recommendation data
        recommendation_data = {
            'Nitrogen': float(data['Nitrogen']),
            'Phosphorus': float(data['Phosphorus']),
            'Potassium': float(data['Potassium']),
            'Ph': float(data['Ph']),
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'Rainfall': float(data['Rainfall'])
        }

        # Log recommendation data for debugging
        logging.info(f"Recommendation data: {recommendation_data}")

        # Prepare features for prediction
        feature_list = [
            recommendation_data['Nitrogen'],
            recommendation_data['Phosphorus'],
            recommendation_data['Potassium'],
            recommendation_data['Temperature'],
            recommendation_data['Humidity'],
            recommendation_data['Ph'],
            recommendation_data['Rainfall']
        ]

        # Log feature list for debugging
        logging.info(f"Feature list for prediction: {feature_list}")

        # Perform prediction
        single_pred = np.array(feature_list).reshape(1, -1)
        logging.info(f"Single prediction array: {single_pred}")

        scaled_features = ms.transform(single_pred)
        logging.info(f"Scaled features: {scaled_features}")

        final_features = sc.transform(scaled_features)
        logging.info(f"Final features: {final_features}")

        prediction = model.predict(final_features)
        logging.info(f"Model prediction: {prediction}")

        # Crop mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Generate recommendation
        crop = crop_dict.get(prediction[0], "Unknown Crop")
        recommendation = {
            'crop': crop,
            'recommendation_text': f"{crop} is the best crop to be cultivated based on the provided soil parameters."
        }

        logging.info(f"Crop Recommendation: {recommendation}")
        return jsonify(recommendation)
    
    except Exception as e:
        logging.error(f"Error in recommendation: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/soil-testing')
def soil_testing():
    return render_template('soil_testing.html')

@app.route('/submit-soil-test', methods=['POST'])
def submit_soil_test():
    try:
        # Get form data
        data = request.json
        logging.info(f"Received soil test request: {data}")
        return jsonify({'message': 'Soil test request received successfully!'})
    except Exception as e:
        logging.error(f"Soil test submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    try:
        # Get form data
        data = request.json
        logging.info(f"Received contact form submission: {data}")
        
        # Optional: Add email sending or database logging logic here
        return jsonify({
            'message': 'Thank you for your message! We will get back to you soon.'
        })
    except Exception as e:
        logging.error(f"Contact form submission error: {e}")
        return jsonify({'error': str(e)}), 500

# Updated route for the crop guide page
@app.route('/crop-guide')
def crop_guide():
    return render_template('guide.html')  # Updated to point to guide.html

if __name__ == "__main__":
    app.run(debug=True)