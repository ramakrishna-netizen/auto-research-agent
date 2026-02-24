from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import os
from agent.graph import build_graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Autonomous Research Agent")

os.makedirs("public", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("public/index.html")

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query")
        if not query:
            await websocket.send_json({"node": "error", "message": "No query provided"})
            return
            
        queue = asyncio.Queue()
        agent_graph = build_graph()
        
        async def run_agent():
            try:
                config = {"configurable": {"queue": queue}}
                final_state = await agent_graph.ainvoke({"query": query, "loop_count": 0}, config)
                await queue.put({
                    "node": "system", 
                    "message": "Task Completed", 
                    "report": final_state.get("report")
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                await queue.put({"node": "error", "message": str(e)})
                await queue.put({
                    "node": "system", 
                    "message": "Task Completed (Error)", 
                    "report": f"An error occurred: {e}"
                })

        # Run agent in background and consume queue in foreground to stream WS
        task = asyncio.create_task(run_agent())
        
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("node") == "system" and msg.get("message").startswith("Task Completed"):
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error handling websocket: {e}")
