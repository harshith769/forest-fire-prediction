import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])
        Region = float(request.form["Region"])

        data = [[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]

        data = scaler.transform(data)

        prediction = ridge_model.predict(data)

        return render_template("index.html", results=prediction[0])

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)