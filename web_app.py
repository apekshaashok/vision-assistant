from flask import Flask, render_template, jsonify, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/web_output")
def web_output():
    try:
        with open("web_output.log", "r") as f:
            data = f.read()
    except FileNotFoundError:
        data = "No output yet!"
    return jsonify({"output": data})

@app.route("/vision_image")
def vision_image():
    # For now, just return the last static image (replace with frame.jpg or any dynamic output)
    return send_file("image.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True, port=5001)

