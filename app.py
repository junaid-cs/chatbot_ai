from flask import Flask, request, jsonify
# from sql_agent import process_input   // You need to create this file and use process_input function
# 
# If you are function from the sql_agent_langraph_final.ipynb file
# from sql_agent_langgraph_final import check_the_given_query

app = Flask(__name__)

# I am able to hit the following route running curl command in terminal,
"""
curl "http://127.0.0.1:5000/search?name=Junaid"

Returns:
    {"result":"You searched for: Junaid"}
"""

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('name')
    return jsonify({'result': f'You searched for: {query}'})

@app.route('/query', methods=["GET", "POST"])
def handle_query():
    query = request.args.get("q") if request.method == "GET" else request.json.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    # response = process_input(query)
    response = {"data": 'response'}
    return jsonify({"result": response})

if __name__ == "__main__":
    app.run(debug=True)  # Disable debug mode for cleaner execution