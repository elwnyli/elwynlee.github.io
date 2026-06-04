# VLM Bridge 使用指南

让 DeepSeek、Kimi 等纯文本模型通过外部视觉模型间接理解图片。

---

## ⚠️ 发给 Claude Code 之前，你需要先准备好

**你必须先有一个视觉模型的 API Key**，三选一：

| 服务商 | 怎么获取 | 免费额度 |
|--------|---------|---------|
| ModelScope（推荐） | 注册 [modelscope.cn](https://modelscope.cn) → 设置 → API Key；**注意还要绑定阿里云账号！** | 有 |
| SiliconFlow | 注册 [siliconflow.cn](https://siliconflow.cn) → 获取 Key | 有 |
| OpenAI | 需要绑卡，Key 是 `sk-` 开头 | 无 |

拿到 Key 之后，把这份文档发给 Claude Code，对它说：

> 帮我按照这份文档配置 VLM Bridge，我的 API Key 是 xxx（替换成你的 Key）

Claude Code 会自动完成所有操作：创建文件夹、保存脚本、填入 Key、注册 Hook。你只需要配合确认即可。

配置完成后，**必须重启 Claude Code** 才会生效。

---

> 这份文档的重点不是”尽可能多地自动发现图片”，而是”只分析当前这轮用户明确提交的图片”。否则很容易把历史截图、旧缓存或最近目录里的无关图片一起送给视觉模型，导致模型回答时混入之前图片的内容。

---

## 目标

Claude Code 的 `UserPromptSubmit` Hook 在用户发消息前执行：

1. 识别当前 prompt 中明确提交的图片。
2. 调用一个 OpenAI-compatible 视觉模型，把图片转成文字描述。
3. 把“本轮图片分析结果”注入 prompt。
4. DeepSeek 等纯文本模型基于这段文字回答。

---

## 设计原则

### 当前轮优先

只分析当前轮用户明确提交的图片：

- prompt 中的 `[Image #N]` 占位符。
- prompt 中明确写出的图片文件路径。
- 当 prompt 明确表达“看这张图 / 截图 / 图片 / 报错”等意图时，读取当前剪贴板图片。

不要默认扫描历史目录、下载目录、桌面目录或整个 `~/.claude/image-cache` 里的最近图片。

### 多图支持

如果当前轮明确提交多张图，就应该一起分析。

例如：

```text
[Image #1] [Image #2] 帮我比较这两张图
```

此时允许分析 2 张图。

但如果历史缓存里也有旧的 `1.png`、`2.png`，不能把旧图一起分析。每个 `[Image #N]` 只取当前时间窗口内最新匹配的一张。

### 禁止误捞历史图片

以下行为默认关闭：

- 扫描桌面、下载目录、截图目录。
- 扫描 `~/.claude/image-cache` 中最近 N 秒内的所有图片。
- 因为没有匹配到图片，就“猜测”用户想分析某张最近图片。

这些 fallback 看似方便，但正是“当前截图 + 之前截图一起被分析”的主要原因。

---

## 工作流程

```text
用户发送消息
    |
    v
UserPromptSubmit Hook
    |
    v
image_preprocess.py
    |
    |-- Path 0: prompt 中有 [Image #N]，只匹配这些编号对应的当前图片
    |-- Path 1: prompt 中有明确图片路径，分析这些路径
    |-- Path 2: prompt 有看图意图时，读取剪贴板中的当前图片
    |-- Path 3: 最近文件扫描，默认关闭
    |
    v
调用视觉模型
    |
    v
注入“Current turn image analysis”
    |
    v
纯文本模型回答
```

---

## 准备工作

- Python 3.9+
- 一个 OpenAI-compatible 视觉模型 API

macOS 读取剪贴板图片需要：

```bash
brew install pngpaste
```

Linux 二选一：

```bash
# Wayland
sudo apt install wl-clipboard

# X11
sudo apt install xclip
```

Windows 使用 PowerShell 读取剪贴板图片，无需额外安装。

---

## 创建脚本

```bash
mkdir -p ~/.claude/vlm-bridge/.claude/hooks
```

将下方完整脚本保存为：

```text
~/.claude/vlm-bridge/.claude/hooks/image_preprocess.py
```

然后执行：

```bash
chmod +x ~/.claude/vlm-bridge/.claude/hooks/image_preprocess.py
```

---

## 配置 API

编辑脚本顶部的 `CONFIG`。

ModelScope 示例：

```python
CONFIG = {
    "api_base": "https://api-inference.modelscope.cn/v1",
    "api_key": "你的 ModelScope API Key",
    "model": "Qwen/Qwen3.5-35B-A3B",
    ...
}
```

OpenAI 示例：

```python
CONFIG = {
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-你的 OpenAI Key",
    "model": "gpt-4o",
    ...
}
```

Ollama 示例：

```python
CONFIG = {
    "api_base": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "gemma3:27b",
    ...
}
```

---

## 注册 Hook

编辑 `~/.claude/settings.json`，在顶层 JSON 中加入：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/vlm-bridge/.claude/hooks/image_preprocess.py",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

如果已有 `hooks` 字段，请合并，不要覆盖已有配置。

---

## 完整脚本

```python
#!/usr/bin/env python3
"""
VLM Bridge — Claude Code UserPromptSubmit Hook

Purpose:
  Let text-only models understand images by converting only the current
  turn's explicit images into text through a vision model.

Important:
  Do not silently analyze historical images. Multi-image support is allowed
  only when the current prompt explicitly references multiple images.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request


CONFIG = {
    "api_base": "https://api-inference.modelscope.cn/v1",
    "api_key": "YOUR_API_KEY_HERE",
    "model": "Qwen/Qwen3.5-35B-A3B",
    "max_image_bytes": 10 * 1024 * 1024,
    "max_images_per_turn": 8,
    "cache_match_max_age_seconds": 120,
    "enable_recent_scan": False,
    "recent_age_seconds": 30,
    "timeout": 120,
    "vision_prompt": (
        "Analyze ONLY the image or images attached in this current request. "
        "Do not use or refer to any previous images, previous screenshots, "
        "or previous image-analysis text. If multiple images are provided, "
        "describe them in the same order as provided, clearly marking Image 1, "
        "Image 2, etc. For each image, describe visible UI, text/OCR, errors, "
        "layout, and important details. Do not invent unseen content. Reply in "
        "the language of the user's request when possible."
    ),
}


SYSTEM = platform.system()
IMAGE_CACHE_DIR = os.path.expanduser("~/.claude/image-cache")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".heic", ".heif"}
MIME = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
    "bmp": "bmp",
}

if SYSTEM == "Windows":
    SCAN_DIRS = [
        os.path.expandvars(r"%USERPROFILE%\Desktop"),
        os.path.expandvars(r"%USERPROFILE%\Downloads"),
        os.path.expandvars(r"%USERPROFILE%\Pictures\Screenshots"),
    ]
else:
    SCAN_DIRS = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads"),
    ]


def _ordered_image_refs(text: str) -> list[str]:
    """Return [Image #N] refs in prompt order, deduplicated."""
    refs = re.findall(r"\[Image #(\d+)\]", text)
    out = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            out.append(ref)
            seen.add(ref)
    return out


def _find_cached_images(text: str) -> list[str]:
    """
    Path 0: match Claude Code [Image #N] placeholders.

    Safety rule:
      For each ref, return at most one image: the newest matching file within
      cache_match_max_age_seconds. This prevents older cached images with the
      same numeric filename from being included.
    """
    refs = _ordered_image_refs(text)
    if not refs or not os.path.isdir(IMAGE_CACHE_DIR):
        return []

    now = time.time()
    max_age = CONFIG["cache_match_max_age_seconds"]
    by_ref: dict[str, tuple[float, str]] = {}

    try:
        for root, _dirs, files in os.walk(IMAGE_CACHE_DIR):
            for filename in files:
                name, ext = os.path.splitext(filename)
                if name not in refs or ext.lower() not in IMG_EXTS:
                    continue

                path = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue

                if now - mtime > max_age:
                    continue

                if name not in by_ref or mtime > by_ref[name][0]:
                    by_ref[name] = (mtime, path)
    except PermissionError:
        return []

    return [by_ref[ref][1] for ref in refs if ref in by_ref]


def _find_paths(text: str) -> list[str]:
    """Path 1: extract explicit image paths mentioned by the user."""
    ext_alt = "|".join(re.escape(ext.lstrip(".")) for ext in IMG_EXTS)
    patterns = [
        rf"""(?:~?/[^\s"'()<>]+\.(?:{ext_alt}))""",
        rf"""(?:[A-Za-z]:\\[^\s"'()<>]+\.(?:{ext_alt}))""",
        rf"""(?:\\\\[^\s"'()<>]+\.(?:{ext_alt}))""",
    ]

    found = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            path = os.path.expanduser(match) if SYSTEM != "Windows" else match
            if os.path.isfile(path) and path not in seen:
                found.append(path)
                seen.add(path)
    return found


def _grab_clipboard() -> str | None:
    """Path 2: grab one image from the system clipboard."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()

    try:
        if SYSTEM == "Darwin":
            result = subprocess.run(["pngpaste", tmp.name], capture_output=True, timeout=10)
            if result.returncode == 0 and os.path.getsize(tmp.name) > 100:
                return tmp.name

        elif SYSTEM == "Windows":
            ps_script = f'''
            Add-Type -AssemblyName System.Windows.Forms,System.Drawing
            $img = [System.Windows.Forms.Clipboard]::GetImage()
            if ($img) {{ $img.Save("{tmp.name}", [System.Drawing.Imaging.ImageFormat]::Png) }}
            '''
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0 and os.path.getsize(tmp.name) > 100:
                return tmp.name

        else:
            for cmd in (
                ["wl-paste", "--type", "image/png"],
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            ):
                try:
                    with open(tmp.name, "wb") as f:
                        result = subprocess.run(
                            cmd,
                            stdout=f,
                            stderr=subprocess.DEVNULL,
                            timeout=10,
                        )
                    if result.returncode == 0 and os.path.getsize(tmp.name) > 100:
                        return tmp.name
                except Exception:
                    continue

    except Exception:
        pass

    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return None


def _scan_recent() -> list[str]:
    """
    Path 3: optional unsafe fallback.

    Disabled by default because it can pick up unrelated screenshots from
    previous turns. Enable only if you accept that tradeoff.
    """
    now = time.time()
    window = CONFIG["recent_age_seconds"]
    hits: list[tuple[float, str]] = []

    for directory in SCAN_DIRS:
        if not os.path.isdir(directory):
            continue
        try:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(filename)[1].lower() not in IMG_EXTS:
                    continue
                age = now - os.path.getmtime(path)
                if 0 < age < window:
                    hits.append((age, path))
        except PermissionError:
            continue

    return [path for _age, path in sorted(hits)]


CLIPBOARD_INTENT_KEYWORDS = [
    "这个",
    "截图",
    "报错",
    "这张",
    "图片",
    "看看",
    "看一下",
    "这是什么",
    "帮我看看",
    "图上",
    "图中",
    "screenshot",
    "clipboard",
    "image",
    "picture",
    "剪贴板",
    "贴图",
]


def _should_try_clipboard(prompt: str) -> bool:
    lower = prompt.lower()
    return any(keyword.lower() in lower for keyword in CLIPBOARD_INTENT_KEYWORDS)


def _limit_images(paths: list[str]) -> list[str]:
    limit = CONFIG["max_images_per_turn"]
    if len(paths) > limit:
        print(
            f"[vlm-bridge] found {len(paths)} images; limiting to {limit}",
            file=sys.stderr,
        )
    return paths[:limit]


def _encode_image(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None

    if size > CONFIG["max_image_bytes"]:
        print(f"[vlm-bridge] SKIP {path} ({size / 1024 / 1024:.1f} MB)", file=sys.stderr)
        return None

    with open(path, "rb") as f:
        data = f.read()

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = MIME.get(ext, "png")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def _call_api(image_paths: list[str]) -> str | None:
    content = []

    for path in image_paths:
        uri = _encode_image(path)
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})

    if not content:
        return None

    content.append({"type": "text", "text": CONFIG["vision_prompt"]})

    payload = {
        "model": CONFIG["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
        "temperature": 0.1,
    }

    url = f"{CONFIG['api_base'].rstrip('/')}/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['api_key']}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=CONFIG["timeout"]) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[vlm-bridge] API error: {exc}", file=sys.stderr)
        return None


def _discover_images(prompt: str) -> list[str]:
    images = _find_cached_images(prompt)
    if images:
        print(f"[vlm-bridge] current prompt image-cache match: {images}", file=sys.stderr)
        return _limit_images(images)

    images = _find_paths(prompt)
    if images:
        print(f"[vlm-bridge] explicit path match: {images}", file=sys.stderr)
        return _limit_images(images)

    if _should_try_clipboard(prompt):
        print("[vlm-bridge] trying clipboard ...", file=sys.stderr)
        clip = _grab_clipboard()
        if clip:
            return [clip]

    if CONFIG.get("enable_recent_scan", False):
        print("[vlm-bridge] scanning recent files; this may include unrelated images ...", file=sys.stderr)
        return _limit_images(_scan_recent())

    return []


def main() -> None:
    prompt = sys.stdin.read().strip()
    print(f"[vlm-bridge] platform: {SYSTEM}", file=sys.stderr)

    images = _discover_images(prompt)
    if not images:
        print(prompt)
        return

    print(f"[vlm-bridge] analyzing {len(images)} current-turn image(s) ...", file=sys.stderr)
    description = _call_api(images)

    if not description:
        print(prompt)
        return

    print(f"[vlm-bridge] done ({len(description)} chars)", file=sys.stderr)
    print(
        f"{prompt}\n\n"
        f"[Current turn image analysis — use this for the current user request only. "
        f"Do not treat previous image-analysis blocks as current images.]\n"
        f"{description}\n"
        f"[End of current turn image analysis]"
    )

    for path in images:
        if path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
```

---

## 给 DeepSeek 的改造要求

如果把这份文档交给 DeepSeek，让它改善已有脚本，请让它严格满足下面几点：

1. 保留多图能力：当前 prompt 明确有多张图时，应按 prompt 中出现顺序分析多张图。
2. 禁止历史串图：不能因为 `~/.claude/image-cache` 中存在旧文件，就把旧图也加入当前轮。
3. `[Image #N]` 匹配规则：每个编号最多匹配一张图，取时间窗口内最新的一张。
4. 最近文件扫描默认关闭：`enable_recent_scan` 必须默认为 `False`。
5. 剪贴板只作为有看图意图时的 fallback，并且通常只返回一张当前剪贴板图片。
6. 注入文本必须标明是“Current turn image analysis”，提醒主模型不要把之前轮次的图片分析当成当前图片。
7. 日志要打印本轮实际分析了几张图、路径来自哪里，方便排查。

---

## 常见问题

### 为什么之前会把历史图片一起分析？

因为旧方案会扫描 `~/.claude/image-cache` 和最近文件目录。Claude Code 的图片缓存可能保留多轮会话里的图片，文件名编号也可能重复。只按文件名或最近修改时间扫描，会把之前的截图误认为当前轮图片。

### 当前同时发送多张图怎么办？

可以。只要当前 prompt 中有多个 `[Image #N]`，脚本会按出现顺序分析多张图。关键是“多图必须来自当前 prompt 的显式引用”，而不是来自目录扫描。

### 为什么默认关闭最近文件扫描？

因为它不可靠。最近文件不等于当前用户提交的图片。截图工具、下载器、浏览器缓存、Claude image-cache 都可能在短时间内产生图片文件。

### 如果拖入图片无法识别怎么办？

先看 stderr 日志中是否出现：

```text
[vlm-bridge] current prompt image-cache match: [...]
```

如果没有，说明 prompt 中可能没有 `[Image #N]`，或 Claude Code 的缓存结构与脚本假设不一致。此时可以临时使用明确路径：

```text
帮我看一下 /Users/paul/Desktop/screenshot.png
```

### 如果我就是想启用最近截图自动识别呢？

可以把：

```python
"enable_recent_scan": False,
```

改成：

```python
"enable_recent_scan": True,
```

但要接受它可能误捞历史图或无关图。更稳的做法仍然是拖入图片、粘贴图片，或给出明确文件路径。
