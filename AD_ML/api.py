from price_predict import price_predict
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict/<int:id>', methods=['GET'])
def predict(id):
    return jsonify(price_predict(id))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)