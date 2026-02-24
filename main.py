from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import os
from agent.graph import build_graph
from agent.db import (
    save_session, list_sessions, get_session_by_id, delete_session,
    verify_token, sign_up, sign_in,
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Autonomous Research Agent")

os.makedirs("public", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

async def get_user_from_request(request: Request) -> dict | None:
    """Extract and verify the Bearer token from the Authorization header."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1]
    return await verify_token(token)


# ──────────────────────────────────────────────
#  Pages
# ──────────────────────────────────────────────

@app.get("/")
async def get_index():
    return FileResponse("public/index.html")


# ──────────────────────────────────────────────
#  Auth endpoints
# ──────────────────────────────────────────────

@app.post("/auth/signup")
async def auth_signup(request: Request):
    body = await request.json()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    if not email or not password:
        return JSONResponse(content={"error": "Email and password are required"}, status_code=400)
    result = await sign_up(email, password)
    if "error" in result:
        return JSONResponse(content=result, status_code=400)
    return JSONResponse(content=result)


@app.post("/auth/login")
async def auth_login(request: Request):
    body = await request.json()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    if not email or not password:
        return JSONResponse(content={"error": "Email and password are required"}, status_code=400)
    result = await sign_in(email, password)
    if "error" in result:
        return JSONResponse(content=result, status_code=401)
    return JSONResponse(content=result)


@app.get("/auth/me")
async def auth_me(request: Request):
    """Verify the current token and return user info."""
    user = await get_user_from_request(request)
    if not user:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    return JSONResponse(content={"user": user})


# ──────────────────────────────────────────────
#  Sessions (auth-guarded)
# ──────────────────────────────────────────────

@app.get("/sessions")
async def get_sessions(request: Request):
    """Return list of sessions for the authenticated user."""
    user = await get_user_from_request(request)
    if not user:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    sessions = await list_sessions(user["id"])
    return JSONResponse(content=sessions)


@app.get("/sessions/{session_id}")
async def get_session(session_id: int, request: Request):
    """Return a specific research session for the authenticated user."""
    user = await get_user_from_request(request)
    if not user:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    session = await get_session_by_id(session_id, user["id"])
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
    return JSONResponse(content=session)


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: int, request: Request):
    user = await get_user_from_request(request)
    if not user:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    
    success = await delete_session(session_id, user["id"])
    if not success:
        return JSONResponse(content={"error": "Failed to delete session"}, status_code=500)
    
    return JSONResponse(content={"status": "success"})



# ──────────────────────────────────────────────
#  WebSocket (auth via initial message)
# ──────────────────────────────────────────────

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query")
        token = data.get("token")

        if not query:
            await websocket.send_json({"node": "error", "message": "No query provided"})
            return

        # Verify auth token
        user = await verify_token(token) if token else None
        if not user:
            await websocket.send_json({"node": "error", "message": "Unauthorized. Please log in."})
            return

        user_id = user["id"]
        queue = asyncio.Queue()
        agent_graph = build_graph()

        async def run_agent():
            try:
                config = {"configurable": {"queue": queue}}
                final_state = await agent_graph.ainvoke({"query": query, "loop_count": 0}, config)
                report = final_state.get("report", "")

                # Persist session to Supabase with user_id
                saved = await save_session(query, report, user_id)
                session_id = saved.get("id")

                await queue.put({
                    "node": "system",
                    "message": "Task Completed",
                    "report": report,
                    "session_id": session_id,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                await queue.put({"node": "error", "message": str(e)})
                await queue.put({
                    "node": "system",
                    "message": "Task Completed (Error)",
                    "report": f"An error occurred: {e}",
                })

        task = asyncio.create_task(run_agent())

        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("node") == "system" and msg.get("message", "").startswith("Task Completed"):
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error handling websocket: {e}")
