import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = {"code": 200, "message": "服务正常运行", "data": {"status": "ok"}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

    def do_POST(self) -> None:
        body = {"code": 405, "message": "只支持 GET 方法"}
        self.send_response(405)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

    def log_message(self, *_args: object) -> None:
        return


def main() -> None:
    print("服务器启动在端口 8080，访问 http://localhost:8080/ping 进行测试")
    HTTPServer(("", 8080), PingHandler).serve_forever()


if __name__ == "__main__":
    main()
