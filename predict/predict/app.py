from flask import Flask, render_template, request
from run import TextPredictionModel

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the home page! To get a prediction go to '/predict'"
    

@app.route('/predict', methods=['GET','POST'])
def predict():
    user_input = request.form['text_input']
    modell = TextPredictionModel.from_artefacts('train/data/artefacts/test/2023-01-05-17-54-07')
    # Use your prediction function to predict the label of the text
    label = model.predict(user_input, top_k=1)
    return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run()
