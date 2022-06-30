from util import predFacesUsingCV2
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predictFaces', methods=['GET', 'POST'])
def predict():
    # Get the image from post request
    recImg = request.json
    # Return Out
    return predFacesUsingCV2(recImg)

# Run Application
if __name__ == '__main__':
    app.run()