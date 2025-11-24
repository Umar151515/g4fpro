import httpx

async_client = httpx.AsyncClient(timeout=None)
sync_client = httpx.Client(timeout=None)