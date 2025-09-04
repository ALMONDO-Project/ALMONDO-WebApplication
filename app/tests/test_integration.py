import unittest
import json
from app import app  # Assuming your Flask app is in app.py

class FlaskIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup code that runs once before all tests
        cls.client = app.test_client()

    def test_generate_graph(self):
        # Test to generate a graph (Erdos-Renyi)
        response = self.client.post('/generate-graph', data={
            'graphType': 'erdos_renyi',
            'nodes': 100,
            'prob': 0.1
        })

        # Assert the response code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert the JSON response contains the correct graph data
        data = json.loads(response.data)
        self.assertIn('nodes', data)
        self.assertIn('links', data)
        self.assertIn('num_nodes', data)
        self.assertIn('num_edges', data)

    def test_upload_edgelist(self):
        # Simulate uploading an edge list file
        with open('test_edgelist.txt', 'w') as f:
            f.write("0 1\n1 2\n2 3")  # Create a sample edge list file

        with open('test_edgelist.txt', 'rb') as f:
            response = self.client.post('/upload-edgelist', data={'file': f})

        # Assert the response code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert the correct number of nodes and edges
        data = json.loads(response.data)
        self.assertEqual(data['num_nodes'], 4)
        self.assertEqual(data['num_edges'], 3)



    def test_unsupported_graph_type(self):
        # Test for unsupported graph type
        response = self.client.post('/generate-graph', data={
            'graphType': 'unsupported_graph',
            'nodes': 100,
            'prob': 0.1
        })

        # Assert the response code is 400 (Bad Request)
        self.assertEqual(response.status_code, 400)

        # Assert error message
        data = json.loads(response.data)
        self.assertEqual(data.get('error'), "Unsupported graph type")

    @classmethod
    def tearDownClass(cls):
        # Cleanup code that runs once after all tests
        pass

if __name__ == '__main__':
    unittest.main()
            