# proxy_server.py — listens for Jetson, forwards to client_proxy_streamer
import asyncio
import websockets

PROXY_PORT = 8763
proxy_clients = set()

async def handler(websocket, path):
    print("[proxy] Jetson connected.")
    proxy_clients.add(websocket)
    try:
        async for message in websocket:
            print("[proxy] forwarding message")
            # forward to all connected listeners
            for client in proxy_clients.copy():
                if client != websocket:
                    await client.send(message)
    except websockets.ConnectionClosed:
        print("[proxy] connection closed.")
    finally:
        proxy_clients.discard(websocket)

if __name__ == "__main__":
    print(f"[proxy] listening on ws://0.0.0.0:{PROXY_PORT}")
    start_server = websockets.serve(handler, "0.0.0.0", PROXY_PORT)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
