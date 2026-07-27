#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
glm_client.py

简单的 GLM Chat API 封装 + 一键自检 API Key 是否可用。

用法：
  1）作为模块导入：
      from glm_client import glm_chat
      resp = glm_chat([{"role": "user", "content": "你好"}])

  2）直接在命令行测试：
      export GLM_API_KEY="your_api_key"
      python glm_client.py
"""

import os
import sys
import json
import requests

BASE_URL = "https://api-gateway.glm.ai/v1"
# 建议：在环境里 `export GLM_API_KEY=xxx`
API_KEY = os.environ.get("GLM_API_KEY")


def glm_chat(messages, model: str = "glm-4-air", temperature: float = 0.0, timeout: int = 120) -> str:
    """
    调用 GLM chat/completions 接口，返回 assistant 的文本内容。

    参数：
        messages   : [{"role": "system"/"user"/"assistant", "content": "..."}]
        model      : 模型名，默认 "glm-4-air"
        temperature: 采样温度
        timeout    : 请求超时时间（秒）
    """
    if API_KEY is None:
        raise RuntimeError(
            "环境变量 GLM_API_KEY 未设置，请先运行：export GLM_API_KEY=your_api_key"
        )

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # 打印更多调试信息
        print("[ERROR] HTTPError:", e, file=sys.stderr)
        print("[ERROR] Response status:", resp.status_code, file=sys.stderr)
        try:
            print("[ERROR] Response body:", resp.text, file=sys.stderr)
        except Exception:
            pass
        raise

    data = resp.json()
    # 也可以在这里 print(json.dumps(data, ensure_ascii=False, indent=2)) 做调试
    return data["choices"][0]["message"]["content"]


def _self_test():
    """简单跑一条 message，测试 API Key 是否可用。"""
    print("=== GLM API Self-test ===")
    if API_KEY is None:
        print("[-] GLM_API_KEY 未设置，请先运行：export GLM_API_KEY=your_api_key")
        sys.exit(1)

    print(f"[INFO] 使用 BASE_URL = {BASE_URL}")
    print("[INFO] 尝试调用 glm_chat()...")

    messages = [
        {
            "role": "user",
            "content": "请用一句话回答：这是一条用于测试 API Key 是否可用的消息。",
        }
    ]

    try:
        reply = glm_chat(messages, model="gemini-3-pro-preview", temperature=0.0)
    except Exception as e:
        print("[-] 调用失败，API Key 可能无效或网络异常。")
        print("    具体错误：", repr(e))
        sys.exit(1)

    print("[+] 调用成功，API Key 可用！")
    print("Assistant 回复预览：")
    # 只打印前 120 个字符，避免太长
    print("----------------------------------------")
    print(reply[:120])
    if len(reply) > 120:
        print("... (截断)")
    print("----------------------------------------")


if __name__ == "__main__":
    _self_test()