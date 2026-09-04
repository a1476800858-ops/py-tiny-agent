import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer

from workspace.ping_server import PingHandler


class PingTest(unittest.TestCase):
    def test_get_and_post_status(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), PingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", "/ping")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["data"]["status"], "ok")
            connection.request("POST", "/ping")
            self.assertEqual(connection.getresponse().status, 405)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
