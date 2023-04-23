import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from detect import *
import uuid

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = './uploads'


def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/')
@cross_origin()
def index():
    args = request.args
    fileid = args.get("fileid")
    print(fileid)
    return render_template('index.html', fileid=f"{fileid}.jpg")


@app.route('/', methods=['POST'])
@cross_origin()
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    id = str(uuid.uuid4())
    detect(os.path.join(app.config['UPLOAD_PATH'], filename), id)
    return id


@app.route('/uploads/<filename>')
@cross_origin()
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/assets/<path:path>')
@cross_origin()
def asset(path):
    return send_from_directory("assets", path)


if __name__ == "__main__":
    app.run(debug=True)
