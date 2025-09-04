import unittest
import json
from app import app
from io import BytesIO

class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_index(self):
        """Test the index route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)

    def test_generate_graph_erdos_renyi(self):
        """Test the /generate-graph route for Erdos-Renyi graph."""
        data = {
            'graphType': 'erdos_renyi',
            'nodes': 10,
            'prob': 0.5
        }
        response = self.client.post('/generate-graph', data=data)
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn('nodes', response_json)
        self.assertIn('links', response_json)
        self.assertEqual(response_json['num_nodes'], 10)

    def test_generate_graph_invalid_type(self):
        """Test the /generate-graph route with invalid graph type."""
        data = {
            'graphType': 'invalid_type',
            'nodes': 10,
            'prob': 0.5
        }
        response = self.client.post('/generate-graph', data=data)
        self.assertEqual(response.status_code, 400)
        response_json = json.loads(response.data)
        self.assertEqual(response_json['error'], "Unsupported graph type")

    def test_upload_edgelist(self):
        """Test the /upload-edgelist route with a file."""
        data = {
            'file': (BytesIO(b'1 2\n2 3\n3 4\n'), 'test.edgelist')
        }
        response = self.client.post('/upload-edgelist', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn('nodes', response_json)
        self.assertIn('links', response_json)


    def test_upload_adjacency_matrix(self):
        """Test the /upload-matrix route with a file."""
        matrix_data = '0,1,0\n1,0,1\n0,1,0\n'
        data = {
            'file': (BytesIO(matrix_data.encode()), 'test.csv')
        }
        response = self.client.post('/upload-matrix', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn('nodes', response_json)
        self.assertIn('links', response_json)

    
if __name__ == '__main__':
    unittest.main()
