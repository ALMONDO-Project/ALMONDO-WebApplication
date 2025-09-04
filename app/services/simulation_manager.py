import uuid

class SimulationManager:
    def __init__(self):
        self.graphs = {}      # session_id -> graph (networkx)
        self.models = {}      # session_id -> model
        self.sim_ids = {}      # session_id -> sim_id

    def create(self, graph): #, model, sim_id):
        session_id = str(uuid.uuid4())
        self.graphs[session_id] = graph
        # self.models[session_id] = model if model else None
        # self.sim_ids[session_id] = sim_id if sim_id else None
        return session_id

    def get_graph(self, session_id):
        return self.graphs.get(session_id)

    def get_model(self, session_id):
        return self.models.get(session_id)
    
    def get_sim_id(self, session_id):
        return self.sim_ids.get(session_id)

    def update_model(self, session_id, model):
        self.models[session_id] = model

    def update_sim_id(self, session_id, sim_id):
        self.sim_ids[session_id] = sim_id

    def delete(self, session_id):
        self.graphs.pop(session_id, None)
        self.models.pop(session_id, None)
        self.sim_ids.pop(session_id, None)

    def list_all(self):
        return list(self.models.keys())