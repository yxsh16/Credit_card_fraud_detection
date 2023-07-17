from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import apply_prediction_pipeline

application = Flask(__name__)
app = application

@app.route('/')
# @cross_origin()
def home_page():
    return render_template('upload_file.html')

@app.route('/upload', methods=['POST'])
# @cross_origin()
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                file_path = 'uploaded.csv'
                file.save(file_path)

                input_file = file_path
                output_file = 'predictions.csv'

                apply_prediction_pipeline(input_file, output_file)

                return jsonify({'success': True, 'message': 'Prediction completed successfully.'})
            except Exception as e:
                return jsonify({'success': False, 'message': 'Error occurred during prediction.', 'error': str(e)})
        else:
            return jsonify({'success': False, 'message': 'No file was uploaded.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
