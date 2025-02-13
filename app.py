from flask import Flask, render_template

import color

app = Flask(__name__)

@app.route('/')
def index():
    return "Nothing important"

@app.route('/colorblocks')
def colors():
    return render_template('index.html')

@app.route('/state')
def state():
    positions = color.get_positions()
    if(positions == None):
        return []
    else:
        return positions

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
