import asyncio
import websockets

async def test_ws_only_connect():
    uri = "ws://192.168.1.7:8000/xiaozhi/v1/"
    try:
        async with websockets.connect(uri, subprotocols=["xiaozhi"]) as websocket:
            print("✅ WebSocket 连接成功")
            await asyncio.sleep(2)
    except Exception as e:
        print("❌ WebSocket 连接失败:", e)

asyncio.run(test_ws_only_connect())
