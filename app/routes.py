from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['GET','POST'])
def enhance():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'audio' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['audio']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('static', filename))
            # Here you would normally call your enhance_audio function
            # For now, we'll just pretend we did and save the original file
            enhanced_audio = filename  # This should be the path to the enhanced audio
            return render_template('index.html', enhanced_audio=enhanced_audio)
    else:
        return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')