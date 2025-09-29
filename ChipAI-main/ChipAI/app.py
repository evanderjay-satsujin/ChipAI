from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import psycopg2
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
from dotenv import load_dotenv
from psycopg2 import OperationalError
from psycopg2.extras import RealDictCursor
from io import BytesIO
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="ChipAI/models/mobilenetv2_finalista.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("TFLite model input details: %s", input_details)

# Ensure uploads directory exists
upload_folder = 'Uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

load_dotenv()

def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            dbname=os.getenv('DB_NAME')
        )
        logger.info("Database connection established.")
        return connection
    except OperationalError as e:
        logger.error("Database connection failed: %s", e)
        raise

# IMAGE_MAPPING
IMAGE_MAPPING = {
    "Siling Labuyo": "siling_labuyo.jpg",
    "Siling Atsal": "bell_pepper.jpg",
    "Siling Espada": "siling_haba.jpg",
    "Scotch Bonnet": "scotch_bonnet.jpg",
    "Siling Talbusan": "siling_talbusan.jpg"
}

# preprocess_image and predict_chili_variety
def preprocess_image(image_stream, target_size=(224, 224)):
    try:
        img = Image.open(image_stream).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logger.info("Image preprocessing successful: shape=%s", img_array.shape)
        return img_array
    except Exception as e:
        logger.error("Error in preprocess_image: %s", str(e))
        return None

def predict_chili_variety(image_stream):
    try:
        img_array = preprocess_image(image_stream, target_size=(224, 224))
        if img_array is None:
            raise ValueError("Image preprocessing failed")
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.info("Prediction raw output (TFLite): %s", output_data)
        class_labels = ["Siling Atsal", "Siling Labuyo", "Siling Espada", "Scotch Bonnet", "Siling Talbusan"]
        predicted_prob = np.max(output_data[0])
        predicted_label = class_labels[np.argmax(output_data[0])]
        confidence = float(predicted_prob)
        if predicted_prob < 0.50:
            logger.info("Prediction below threshold: %s (confidence: %.4f)", predicted_label, confidence)
            return {"label": "No Chili Detected", "confidence": confidence}
        logger.info("Prediction result: %s (confidence: %.4f)", predicted_label, confidence)
        return {"label": predicted_label, "confidence": confidence}
    except Exception as e:
        logger.error("Error in TFLite prediction: %s", str(e))
        return {"label": "Error processing the image", "error": str(e)}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        
        if not username or not password or not confirm_password:
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match.'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Check if username exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                logger.warning("Signup failed: Username %s already exists", username)
                return jsonify({'success': False, 'message': 'Username already exists.'}), 400

            # Call sp_signup
            cursor.execute("CALL sp_signup(%s, %s)", (username, password))
            conn.commit()

            # Get the inserted user's ID
            cursor.execute("SELECT id, is_admin FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            if not user:
                logger.error("sp_signup failed: User %s not found after insertion", username)
                return jsonify({'success': False, 'message': 'Signup failed: User not created.'}), 500

            session['user_id'] = user['id']
            is_admin = user['is_admin']
            logger.info("User signed up and logged in: %s, is_admin: %s", username, is_admin)
            return jsonify({'success': True, 'message': 'Signup successful', 'is_admin': is_admin}), 200

        except psycopg2.Error as e:
            conn.rollback()
            logger.error("Database error during signup: %s", str(e))
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
        except Exception as e:
            conn.rollback()
            logger.error("Unexpected error during signup: %s", str(e))
            return jsonify({'success': False, 'message': f'Unexpected error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required.'}), 400
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT * FROM sp_login(%s, %s)", (username, password))
            result = cursor.fetchone()
            logger.info("sp_login result for %s: %s", username, result)
            
            if not result:
                logger.warning("Login failed: No result from sp_login for %s", username)
                return jsonify({'success': False, 'message': 'User not found.'}), 401

            user_id = result.get('user_id')
            status = result.get('status')

            if status == 'User authenticated' and user_id is not None:
                session['user_id'] = user_id
                cursor.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
                is_admin = user['is_admin'] if user else False
                logger.info("User logged in: %s, is_admin: %s", username, is_admin)
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'is_admin': is_admin
                }), 200
            else:
                logger.warning("Login failed for %s: status=%s", username, status)
                return jsonify({'success': False, 'message': status}), 401
        except psycopg2.Error as e:
            logger.error("Database error during login: %s", str(e))
            return jsonify({'success': False, 'message': f"Database error: {str(e)}"}), 500
        except Exception as e:
            logger.error("Unexpected error during login: %s", str(e))
            return jsonify({'success': False, 'message': f"Unexpected error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
    if 'user_id' not in session:
        logger.info("User not logged in, rendering login page")
        return render_template('index.html')
    logger.info("User already logged in, redirecting to dashboard")
    return redirect(url_for('dashboard'))

@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    if 'user_id' not in session:
        logger.info("No user_id in session, redirecting to login")
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT is_admin FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        if not user or not user['is_admin']:
            logger.info("Non-admin user attempted to access admin_dashboard")
            return redirect(url_for('dashboard'))
        
        cursor.execute("""
            SELECT u.username, f.prediction, f.feedback_text, f.timestamp
            FROM feedback f
            LEFT JOIN users u ON f.user_id = u.id
            ORDER BY f.timestamp DESC
        """)
        feedback_data = cursor.fetchall()
        
        for record in feedback_data:
            if not record['username']:
                logger.warning("Feedback ID %s has no matching username (user_id: %s)", record.get('id'), record.get('user_id'))
        
        return render_template('admin_dashboard.html', feedback_data=feedback_data)
    except Exception as e:
        logger.error("Error fetching admin dashboard data: %s", str(e))
        return jsonify({'error': 'Failed to load dashboard'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/create_admin', methods=['POST'])
def create_admin():
    if 'user_id' not in session:
        logger.info("No user_id in session for create_admin")
        return jsonify({'success': False, 'message': 'User not logged in.'}), 401
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT is_admin FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        if not user or not user['is_admin']:
            logger.info("Non-admin user attempted to create admin account")
            return jsonify({'success': False, 'message': 'Unauthorized: Admin access required.'}), 403

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not username or not password or not confirm_password:
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match.'}), 400

        cursor.execute("CALL sp_signup(%s, %s)", (username, password))
        cursor.execute("UPDATE users SET is_admin = TRUE WHERE username = %s", (username,))
        conn.commit()
        logger.info("Admin account created: %s", username)
        return jsonify({'success': True, 'message': 'Admin account created successfully.'}), 200
    except psycopg2.Error as e:
        conn.rollback()
        logger.error("Database error creating admin account: %s", str(e))
        return jsonify({'success': False, 'message': f"Database error: {str(e)}"}), 500
    except Exception as e:
        conn.rollback()
        logger.error("Unexpected error creating admin account: %s", str(e))
        return jsonify({'success': False, 'message': f"Unexpected error: {str(e)}"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/get_username', methods=['GET'])
def get_username():
    if 'user_id' not in session:
        logger.info("No user_id in session for get_username")
        return jsonify({'error': 'User not logged in'}), 401
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT username FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        return jsonify({'username': user['username']})
    except Exception as e:
        logger.error("Error fetching username: %s", str(e))
        return jsonify({'error': 'Failed to fetch username'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        logger.info("No user_id in session for submit_feedback")
        return jsonify({'error': 'User not logged in'}), 401
    data = request.get_json()
    user_id = session['user_id']
    prediction = data.get('prediction')
    feedback_text = data.get('feedback_text', '')
    if not prediction:
        return jsonify({'error': 'Prediction is required'}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO feedback (user_id, prediction, feedback_text, timestamp)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (user_id, prediction, feedback_text)
        )
        conn.commit()
        logger.info("Feedback saved for user_id: %s, prediction: %s", user_id, prediction)
        return jsonify({'success': True}), 200
    except Exception as e:
        conn.rollback()
        logger.error("Error saving feedback: %s", str(e))
        return jsonify({'error': 'Failed to save feedback'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        logger.info("No user_id in session, redirecting to login")
        return redirect(url_for('login'))
    logger.info("Rendering dashboard for user_id: %s", session['user_id'])
    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = image.read()
        image_stream = BytesIO(image_bytes)
        img = Image.open(image_stream)
        img.verify()
        logger.info("Image is valid.")
        image_stream.seek(0)
        result = predict_chili_variety(image_stream)
        if "error" in result:
            return jsonify({'error': f"Failed to process image: {result['error']}"}), 400
        return jsonify({
            'prediction': result['label'],
            'confidence': result['confidence']
        })
    except (IOError, SyntaxError) as e:
        logger.error("Invalid image file: %s", str(e))
        return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400
    except Exception as e:
        logger.error("Error processing the image: %s", str(e))
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

@app.route('/')
def index():
    if 'user_id' in session:
        logger.info("User logged in, redirecting to dashboard")
        return redirect(url_for('dashboard'))
    logger.info("User not logged in, rendering index.html")
    return render_template('index.html')

@app.route('/ai')
def ai_model():
    if 'user_id' not in session:
        logger.info("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    return render_template('AI.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/FAQs')
def faqs():
    if 'user_id' not in session:
        logger.info("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    return render_template('faqs.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    success = True
    if success:
        flash('User added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    logger.info("User logged out")
    return redirect(url_for('index'))

@app.route('/get_chili_info', methods=['GET'])
def get_chili_info():
    chili_name = request.args.get('name')
    if not chili_name or chili_name in ["Error processing the image", "No Chili Detected"]:
        return jsonify({'error': 'Invalid chili name'}), 400
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT name, english_name, scientific_name, shu_range, description
            FROM chili_varieties WHERE name = %s
        """, (chili_name,))
        chili_info = cursor.fetchone()
        if not chili_info:
            return jsonify({'error': 'Chili not found'}), 404
        chili_info['image_url'] = url_for('static', filename=f'images/{IMAGE_MAPPING.get(chili_name, "default.jpg")}', _external=True)
        return jsonify(chili_info)
    finally:
        cursor.close()
        conn.close()

@app.route('/chili_trivia', methods=['GET'])
def chili_trivia():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT trivia_text FROM chili_trivia ORDER BY RANDOM() LIMIT 1")
        trivia = cursor.fetchone()
        if not trivia:
            return jsonify({'error': 'No trivia available'}), 404
        return jsonify(trivia)
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))