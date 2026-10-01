import httpx
from supabase import create_client, Client, ClientOptions
from app.core.config import settings

supabase_client: Client = None

if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    # Use HTTP/1.1 and a longer timeout to prevent HTTP/2 proxy drops and request timeouts
    sync_client = httpx.Client(http2=False, timeout=60.0)
    options = ClientOptions(httpx_client=sync_client)
    supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY, options=options)
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not configured. Supabase operations will fail.")
