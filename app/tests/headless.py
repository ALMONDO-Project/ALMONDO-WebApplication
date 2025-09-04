import random
import os
import tempfile
import pytest
import networkx as nx
from flask import Flask, request, jsonify, send_file, session
from flask.testing import FlaskClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['SECRET_KEY'] = 'your_secret_key'


# Graph generation route
@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    graphType = request.form.get('graphType')
    N = int(request.form.get('nodes'))
    prob = float(request.form.get('prob'))
    
    if graphType == 'erdos_renyi':
        G = nx.erdos_renyi_graph(N, prob)
        G.graph['type'] = 'erdos_renyi'
    
    nodes = [{"id": str(node)} for node in G.nodes()]
    links = [{"source": str(edge[0]), "target": str(edge[1])} for edge in G.edges()]
    
    return jsonify({"nodes": nodes, "links": links, "num_nodes": len(G.nodes()), "num_edges": len(G.edges())})


# Simulation route
@app.route('/run-simulation', methods=['POST'])
def run_simulation():
    data = request.form
    results = [{
        'iteration': i + 1,
        'phi': float(data['phiValue']) * random.random(),
        'lambda': float(data['lambdaValue']) * random.random(),
        'po': float(data['po']),
        'pp': float(data['pp']),
    } for i in range(int(data['iterations']))]
    
    return jsonify({"status": "Simulation complete", "results": results})


# File upload route
@app.route('/upload-edgelist', methods=['POST'])
def upload_edgelist():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        G = nx.read_edgelist(filepath, delimiter=' ', nodetype=int)
        nodes = [{"id": str(node)} for node in G.nodes()]
        links = [{"source": str(edge[0]), "target": str(edge[1])} for edge in G.edges()]
        return jsonify({"nodes": nodes, "links": links, "num_nodes": len(G.nodes()), "num_edges": len(G.edges())})
    return jsonify({"error": "No file uploaded"}), 400


# File download route
@app.route('/download-edge-list', methods=['GET'])
def download_edge_list():
    if 'graph_filename' in session:
        return send_file(session['graph_filename'], as_attachment=True)
    
    G = nx.erdos_renyi_graph(10, 0.5)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.edgelist')
    nx.write_edgelist(G, temp_file.name)
    session['graph_filename'] = temp_file.name
    return send_file(temp_file.name, as_attachment=True)


# Test fixtures and test cases
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client



def test_generate_graph_erdos_renyi(client: FlaskClient):
    data = {'graphType': 'erdos_renyi', 'nodes': 10, 'prob': 0.5}
    response = client.post('/generate-graph', data=data)
    assert response.status_code == 200
    graph_data = response.get_json()
    assert 'nodes' in graph_data and graph_data['num_nodes'] == 10


def test_run_simulation(client: FlaskClient):
    simulation_data = {
        'initialStatus': 'uniform', 'po': 0.01, 'pp': 0.99,
        'phiValue': 0.1, 'lambdaValue': 0.5, 'iterations': 100,
        'runSimulationOption': 'iteration-bunch',
    }
    response = client.post('/run-simulation', data=simulation_data)
    assert response.status_code == 200
    simulation_data = response.get_json()
    assert simulation_data['status'] == 'Simulation complete'
    assert len(simulation_data['results']) == 100


def test_upload_edgelist(client: FlaskClient):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"1 2\n2 3\n3 4\n")
        tmp_file.close()

        data = {'file': (open(tmp_file.name, 'rb'), 'test_edgelist.txt')}
        response = client.post('/upload-edgelist', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['num_nodes'] == 4 and json_data['num_edges'] == 3


def test_download_edgelist(client: FlaskClient):
    response = client.post('/generate-graph', data={'graphType': 'erdos_renyi', 'nodes': 10, 'prob': 0.5})
    assert response.status_code == 200
    graph_data = response.get_json()
    response = client.get('/download-edge-list')
    assert response.status_code == 200
    assert response.content_type == 'application/octet-stream'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edgelist') as tmp_file:
        tmp_file.write(response.data)
        tmp_file.close()
        with open(tmp_file.name, 'r') as file:
            edges = file.readlines()
            assert len(edges) > 0 
            assert edges[0].startswith("0")  
