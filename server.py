"""迁移自未完成的 server.go 的独立鉴权占位示例。"""


def main(user: object | None = None) -> None:
    print("Server is starting on port 8080...")
    if user is None:
        print("Forbidden!")
        return
    print("Authorized request.")


if __name__ == "__main__":
    main()
