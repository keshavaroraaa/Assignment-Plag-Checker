from flask import Flask, render_template, request
import os
from plag_predictor import tokenize, score_single_file, read_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    report = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            
            # Use your existing logic
            text = read_file(path)
            tokens = tokenize(text)
            report = score_single_file(text, tokens)
            report['filename'] = file.filename
            
    return render_template('index.html', report=report)

if __name__ == '__main__':
    app.run(debug=True)