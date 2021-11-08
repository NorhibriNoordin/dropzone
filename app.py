import os

from flask import Flask, render_template, request, jsonify
from flask_dropzone import Dropzone
import requests
import pandas as pd

# variable declare
basedir = os.path.abspath(os.path.dirname(__file__))

dataset_path = ""

app = Flask(__name__)

target_csv_path = "input.csv"

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_MAX_FILE_SIZE=300,
    DROPZONE_MAX_FILES=1,
    # allow only excel file(xlsx)
    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='.xlsx,.xls,.csv'
)


dropzone = Dropzone(app)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)

        dataset_path = file_path

        f.save(file_path)

        data = pd.read_csv(file_path)

        # data.to_csv(target_csv_path)

        # print("Download ready.")

        # nba = pd.read_csv("input.csv")

        no_index = len(data.index)
        no_columns = len(data.columns)

        supervised = True

        for col in data.columns:
            for character in col:
                if character.isdigit():
                    supervised = False

        if (supervised):
            for col in data.columns:
                print(col)
        else:
            print("Your data is unsupervised")

        if (no_index <= no_columns):
            print("Your data is insufficient for training. Opt for high bias/low variance like Linear regression, Naïve Bayes, or Linear SVM")
        else:
            print("Your data is sufficient for training. Opt for low bias/high variance algorithms like KNN, Decision trees, or kernel SVM")

    # You can return a JSON response then get it on client side:
    # (see template index.html for client implementation)
    # return jsonify(uploaded_path=file_path)

    return render_template('index.html')


@app.route('/finish')
def process():

    from website import process

    process.process_download()

    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
