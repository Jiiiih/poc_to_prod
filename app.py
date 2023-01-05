from flask import Flask, render_template, request
from predict.predict.run import TextPredictionModel
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the home page! To get a prediction go to '/predict'"
    

@app.route("/predict", methods=['POST'])
def predict():
    body=json.loads(request.get_data())
    text_list = body['textsToPredict']
    top_k = body['top_k']
    textPredictionModel = TextPredictionModel.from_artefacts('./path')
    label_list = textPredictionModel.predict(text_list, top_k=top_k)
    print(label_list)
    return str(label_list)
if __name__ == '__main__':
    app.run()
