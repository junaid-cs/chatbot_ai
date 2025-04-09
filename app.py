from flask import Flask, request, jsonify
from sql_agent import process_input  

app = Flask(__name__)

@app.route('/query', methods=["GET", "POST"])
def handle_query():
    query = request.args.get("q") if request.method == "GET" else request.json.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    response = process_input(query)
    return jsonify({"result": response})

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode for cleaner execution