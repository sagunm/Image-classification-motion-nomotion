from flask import Flask

app = Flask(__name__)
@app.route('/practice')

def running():
    return 'Flask is running'
    