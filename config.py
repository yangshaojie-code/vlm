"""全局配置:模型名、接口地址、API Key 读取。"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_ROOT, ".env")


def _load_dotenv(path: str = _ENV_FILE) -> None:
    """把 .env 中 KEY=VALUE 写入 os.environ(不覆盖已有环境变量)。"""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

# 百炼的 OpenAI 兼容接口(通用域名)。如需切换业务空间专属域名,
# 设置环境变量 DASHSCOPE_BASE_URL,例如
# https://llm-ieblg8jrbniinxer.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 可通过环境变量 QWEN_VL_MODEL 覆盖,默认使用旗舰多模态模型 qwen3.7-plus
MODEL_NAME = os.environ.get("QWEN_VL_MODEL", "qwen3.7-plus")

# qwen3.7 系列是混合思考模型且默认开启思考;本项目输出确定性 JSON,
# 关闭思考以降低延迟和 token 消耗(非 OpenAI 标准参数,需经 extra_body 传入)
EXTRA_BODY = {"enable_thinking": False}


def get_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY。\n"
            "推荐(PyCharm 友好):在 vlm_pipeline\\.env 文件中写入一行:\n"
            "  DASHSCOPE_API_KEY=sk-xxxx\n"
            "或设置系统环境变量后完全重启 PyCharm:\n"
            '  PowerShell: setx DASHSCOPE_API_KEY "sk-xxxx"'
        )
    return key


def get_client():
    """返回指向 DashScope 兼容接口的 OpenAI 客户端。"""
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=BASE_URL)
