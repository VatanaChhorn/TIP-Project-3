from flask import Flask, render_template
from flask_cors import CORS
from routes import routes

app = Flask(__name__)
app.secret_key = 'admin' # Secret key for session management
CORS(app)
app.register_blueprint(routes)

# ✅ Serve HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/navbar.html')
def navbar():
    return render_template('navbar.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

# ✅ Run the server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
