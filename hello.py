#import os
#from flask import Flask
#import cv2 as cv
from flask import Flask
from flask import Flask, render_template

from flask import request
from flask import jsonify


#project_root = os.path.dirname(__file__)
#template_path = os.path.join(project_root, 'app/templates')
#app = Flask(__name__, template_folder=template_path)

app = Flask(__name__)

@app.route("/web")
def Web():
    # data = 5 >>> Send to html
    age = 10
    return render_template('index.html', data = age)

@app.route("/student")
def Hi():
    return "Hi my is pon"

@app.route("/")
def hello():
    return "Hello World from Flask PON"

@app.route("/model",methods=['POST'])
def model():
    message = request.get_json(force=True)
    name = message['name']
    response = { 'greeting' : 'Hello, ' + name + '!'}
    return jsonify(response)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(debug=True)
