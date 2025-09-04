from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    
    @task(1)
    def load_homepage(self):
        """Test the response time of the home page"""
        self.client.get("/")
    
    @task(2)
    def generate_graph(self):
        """Simulate user generating a graph"""
        payload = {
            "graph_type": "erdos_renyi",
            "nodes": 100,
            "prob": 0.2
        }
        self.client.post("/generate_graph", json=payload)
    
    @task(3)
    def run_simulation(self):
        """Simulate user clicking the Run Simulation button"""
        payload = {
            "initial_status_option": "uniform",
            "min_range_uniform_distribution": 0.2,
            "max_range_uniform_distribution": 0.8
        }
        self.client.post("/run_simulation", json=payload)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # Simulates user waiting 1-3 seconds be

