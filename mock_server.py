from aiohttp import web
import asyncio
import time

async def health_check(request):
    return web.json_response({"status": "ok"})

async def file_parse(request):
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Return a dummy response
    return web.json_response({
        "backend": "mock-backend",
        "version": "1.0.0",
        "results": {"mock.pdf": {"text": "mock content"}}
    })

app = web.Application()
# MinerU docs health check endpoint
app.router.add_get('/docs', health_check) 
app.router.add_post('/file_parse', file_parse)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    web.run_app(app, host=args.host, port=args.port)
