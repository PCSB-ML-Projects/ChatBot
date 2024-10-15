from flask import Flask, request, jsonify
from llm import return_ans
#importing the function to query the model

from json import loads, dump
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/ask', methods=["POST"])
def ask():
    print("asking llm")
    data = request.get_json()        #getting the data from the request
    print("data extracted successfully")
    print(data)
    query = str(data["content"])     #getting the query from the data
    print(query)
    try:
        response = return_ans(query) #querying the model
        response = {
            "data" : response,
            "status":200,
            "message":"Success"
        }
        return jsonify(response)
    except Exception as e:
        print(e)
        response = {
            "data": e,
            "status": 500,
            "message": "Internal Server Error"
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)