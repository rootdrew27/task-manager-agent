from uuid import uuid1
from PySide6.QtCore import QObject, Signal, Slot, QMutex, QWaitCondition
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command

from agent import graph, task_manager

class TaskManagerAgent(QObject):
    need_human_input = Signal(str)
    agent_done = Signal(str, bool) # Task state and final message

    def __init__(self):
        super().__init__()
        self.graph = graph
        self.task_manager = task_manager
        self.config: RunnableConfig | None

        self.is_awaiting_human_response: bool = False
        
    @Slot(str)
    def route_request(self, transcription: str):
        if self.is_awaiting_human_response:
            self.resume_agent(transcription)

        else: # initial request
            self.config = {"configurable": {"thread_id": uuid1()}}
            self.run_agent({"user_input": transcription})
    
    def run_agent(self, input):
        events = self.graph.stream(input, self.config, stream_mode="values")
        for event in events:
            print(f"EVENT: {event}")
            #if "messages" in event:
             #   print(event["messages"][-1])

        assert self.config is not None, "Configuration must be set."
        snapshot = self.graph.get_state(self.config) 
        has_interrupts = False
        if snapshot.interrupts and len(snapshot.interrupts) > 0:
            assert len(snapshot.interrupts) == 1, "Only one interrupt at a time!"
            has_interrupts = True
            for interrupt in snapshot.interrupts:
                self.need_human_input.emit(interrupt.value)

        if has_interrupts:
            self.is_awaiting_human_response = True
            return
        else:
            self.agent_done.emit(snapshot.values.get("messages", [{"content": "No message"}])[-1].content, True) # agent finished

    def resume_agent(self, input: str):
        assert isinstance(input, str)
        assert self.config is not None
        for event in self.graph.stream(Command(resume=input), self.config, stream_mode="values"):
            print(event)
        self.is_awaiting_human_response = False
        self.agent_done.emit(self.graph.get_state(self.config).values.get("messages", [{"content": "No message"}])[-1].content, True)
        