from concurrent.futures import ThreadPoolExecutor


def count_concurrently(workers: int = 1000) -> int:
    count = 0

    def increment() -> int:
        return 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        count += sum(pool.map(lambda _: increment(), range(workers)))
    return count


if __name__ == "__main__":
    print(f"最终的 Count 是: {count_concurrently()}")
