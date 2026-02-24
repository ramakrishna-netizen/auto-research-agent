import os
import asyncio
from supabase import create_client, Client


def get_supabase() -> Client:
    """Client with anon key — used for auth operations."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file")
    return create_client(url, key)


def get_supabase_admin() -> Client:
    """Client with service_role key — bypasses RLS for server-side DB operations."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY"))
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file")
    return create_client(url, key)



def _extract_data(response):
    """Support dict responses and objects with a .data attribute."""
    if response is None:
        return None
    if isinstance(response, dict):
        return response.get("data")
    if hasattr(response, "data"):
        return response.data
    return None


# ──────────────────────────────────────────────
#  Auth helpers
# ──────────────────────────────────────────────

async def verify_token(token: str) -> dict | None:
    """Validate a Supabase access token and return the user dict.

    Returns the user dict on success, or None on failure.
    """
    try:
        def _verify():
            sb = get_supabase()
            return sb.auth.get_user(token)

        response = await asyncio.to_thread(_verify)
        if response and hasattr(response, "user") and response.user:
            return {"id": response.user.id, "email": response.user.email}
        return None
    except Exception as e:
        print(f"[AUTH] Token verification failed: {e}")
        return None


async def sign_up(email: str, password: str) -> dict:
    """Register a new user with email and password."""
    try:
        def _signup():
            sb = get_supabase()
            return sb.auth.sign_up({"email": email, "password": password})

        response = await asyncio.to_thread(_signup)
        if response and hasattr(response, "user") and response.user:
            session = response.session
            # If Supabase returns a session (email confirmation disabled), use it
            if session and session.access_token:
                return {
                    "user": {"id": response.user.id, "email": response.user.email},
                    "access_token": session.access_token,
                    "refresh_token": session.refresh_token,
                }
            # Otherwise, auto-login immediately after signup
            return await sign_in(email, password)
        return {"error": "Signup failed"}
    except Exception as e:
        print(f"[AUTH] Signup failed: {e}")
        return {"error": str(e)}


async def sign_in(email: str, password: str) -> dict:
    """Sign in an existing user with email and password."""
    try:
        def _signin():
            sb = get_supabase()
            return sb.auth.sign_in_with_password({"email": email, "password": password})

        response = await asyncio.to_thread(_signin)
        if response and hasattr(response, "user") and response.user:
            session = response.session
            return {
                "user": {"id": response.user.id, "email": response.user.email},
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None,
            }
        return {"error": "Invalid credentials"}
    except Exception as e:
        print(f"[AUTH] Signin failed: {e}")
        return {"error": str(e)}


# ──────────────────────────────────────────────
#  Session CRUD (user-scoped)
# ──────────────────────────────────────────────

async def save_session(query: str, report: str, user_id: str) -> dict:
    """Save a completed research session to Supabase, linked to a user."""
    try:
        def _save():
            sb = get_supabase_admin()
            return sb.table("research_sessions").insert({
                "query": query,
                "report": report,
                "user_id": user_id,
            }).execute()

        response = await asyncio.to_thread(_save)
        data = _extract_data(response)
        if data:
            return data[0] if isinstance(data, list) else data
        return {}
    except Exception as e:
        print(f"[DB] Failed to save session: {e}")
        return {}


async def list_sessions(user_id: str) -> list:
    """List research sessions for a specific user, newest first."""
    try:
        def _list():
            sb = get_supabase_admin()
            return (
                sb.table("research_sessions")
                .select("id, query, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )

        response = await asyncio.to_thread(_list)
        data = _extract_data(response)
        return data or []
    except Exception as e:
        print(f"[DB] Failed to list sessions: {e}")
        return []


async def get_session_by_id(session_id: int, user_id: str) -> dict:
    """Retrieve a specific session by ID, scoped to the user."""
    try:
        def _get():
            sb = get_supabase_admin()
            return (
                sb.table("research_sessions")
                .select("*")
                .eq("id", session_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )

        response = await asyncio.to_thread(_get)
        data = _extract_data(response)
        return data or {}
    except Exception as e:
        print(f"[DB] Failed to get session {session_id}: {e}")
        return {}


async def delete_session(session_id: int, user_id: str) -> bool:
    """Delete a specific session, scoped to the user."""
    try:
        def _delete():
            sb = get_supabase_admin()
            return (
                sb.table("research_sessions")
                .delete()
                .eq("id", session_id)
                .eq("user_id", user_id)
                .execute()
            )

        await asyncio.to_thread(_delete)
        return True
    except Exception as e:
        print(f"[DB] Failed to delete session {session_id}: {e}")
        return False

