from PySide6.QtCore import QObject, Signal
from agent import graph, tasks

class TaskManager(QObject):

    agent_done = Signal(dict)

    def __init__(self):
        super().__init__()
        self.graph = graph
        self.tasks = tasks
        
    def process_input(self, transcription_input):
        events = self.graph.stream({"user_input": transcription_input}, stream_mode="updates")
        for event in events:
            print(event)
        self.agent_done.emit(self.tasks)        
