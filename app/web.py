#!/usr/bin/env python
import os
from os import path
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
from resources.app import App

# define where the upload folder will be
UPLOAD_FOLDER = './uploads'
# define the allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# initialize the server configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
app.secret_key = "super secret key"

# create the uploads folder if doesn't exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    ''''
    Check if the file path is allowed to be used
    Args:
        filename: file name and path of the image
    Return:
        Returns True or False if the file is allow to be use by the app
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index_page():
    '''
    Render the index.html file
    '''
    return render_template('index.html', error=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    '''
    Receives the upload action coming from the webpage.
    It Verifies if the files is present to redirect to the predictor.html, otherwise it will show an error.
    '''
    file = request.files['file']

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return render_template('index.html', error="You forgot to select the image")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        new_filename = get_new_filename(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
        return redirect(url_for('predictor', filename=new_filename))

    return render_template('index.html', error=None)


@app.route('/predictor')
def predictor():
    ''''
    Receives the file to be classified. It translates to another unique name.
    '''
    filename = request.args.get('filename')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    message = App().predict(os.path.abspath(image_path))
    return render_template('prediction.html', label=message, filename=filename)


@app.route('/preview_predicted_image/<filename>')
def preview_predicted_image(filename):
    '''
    Result on the image to preview on the predictor.html
    '''
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def get_new_filename(filename):
    '''
    Convertes the file name which comes from the request to another one adding the timestamp
    Args:
        filename: original file name
    Return:
        new filename format, original_filename_yyyy-mm-dd_hh-mm-ss-ms.orignal_extension
    '''
    now = datetime.now()
    date_formatted = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename, file_extension = os.path.splitext(filename)
    new_filename = '{}_{}{}'.format(filename, date_formatted, file_extension)
    print(">>> Converting file name from{} to {}".format(filename, new_filename))
    return new_filename


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

    # TODO SFS force 80 to run on aws, run as sudo
    # sudo nohup python3 web.py &
    # app.run(host='0.0.0.0', port=80)