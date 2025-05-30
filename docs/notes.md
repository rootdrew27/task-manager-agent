# Notes

- Consider using an initial router that determines whether or not the request pertains to task management.
    - If True: pass the request to a tool calling model that uses a SemanticSimilarityExampleSelector.
    - If False: concisely explain to the user why the request is invalid.

## User Inputs

### Get
- What are my tasks?


### Add

#### Need More Info
- I'd like to create a task.
- I'd like to add a task.
- Make a new task.
Tool: **more_info**

#### Add the task
- Add a new task: clean my room.
- New task: oil change.
Tool: **create_task**
Req: Task Name

### Edit

#### Need More Info
- I'd like to update a task.
- Update a task.
- Edit one of my tasks.
Tool: **more_info**

#### Edit the task
- Mark my 'clean room' task as complete.
- Change my 'clean room' task to 'clean bathroom'.
Tool: **Update Task** 
Reqs: Task Name(s) OR description 

### Delete

#### Need more info
- Delete one of my tasks
- 


### Bad Requests
- When has Henry the 8th born?
- Why is the sky blue?
Tool: **reject_request** 