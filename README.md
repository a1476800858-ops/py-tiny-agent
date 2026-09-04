# py-tiny-claw

基础本地工具只使用标准库；真实 LLM 和飞书连接通过可选依赖启用。

## 安装

```text
python -m pip install -e .
python -m pip install -e ".[llm]"
```

## CLI

```text
python -m tiny_claw --prompt "请检查当前项目" --dir . --session cli_default_session
```

默认使用智谱 OpenAI 兼容接口和 `glm-4.5-air`，需要设置 `ZHIPU_API_KEY`。

## 测试

```text
python -m unittest discover -s tests -v
```
