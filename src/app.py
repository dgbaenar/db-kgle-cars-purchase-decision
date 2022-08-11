from flask import Flask, request
import joblib

from model import predict_buy


app = Flask(__name__)

# Load model settings
model = joblib.load('./data/joblib/model.joblib.dat')
imputer = joblib.load('./data/joblib/imputer.joblib.dat')


@app.route('/', methods=['GET'])
def health():
    return 'ok'


@app.route('/predict', methods=['POST'])
def predict():
    input = request.get_json()
    if not input:
        return {
            'error': 'Body is empty.'
        }, 400
    response = predict_buy(input,
                           model,
                           imputer,
                           0.5)
    return response, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, use_reloader=True)
