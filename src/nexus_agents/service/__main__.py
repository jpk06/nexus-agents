"""Process entry point: ``python -m nexus_agents.service``.

三个刻意的选择
--------------
1. **单进程、单 worker**。``uvicorn.run(app, workers=N)`` 需要 app 是可导入
   的字符串；而更根本的问题是：会话状态（记忆、消息流水、串行锁）现在都在
   进程内存里，第二个 worker 会看到一个空的会话表，用户第二句话就 404。
   在多 worker 之前，``SessionStore`` 必须先换成 Redis/Postgres —— 顺序不
   能反。把这件事写在启动代码里，是为了让它成为**决定**而不是隐患。
2. **uvicorn 懒导入**。测试环境只装 ``.[dev]``（有 fastapi，没有 uvicorn），
   而 ``--check`` 这条路径不需要服务器。顶层 import 会让「只想校验配置」
   也得装整套服务器依赖。
3. **配置错误退出码 2**。1 留给「运行中失败」，2 表示「用法/配置错误」。
   容器编排器据此区分「部署参数错了」和「服务崩了」。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..errors import MissingCredentialsError
from .app import create_app
from .runtime import EnvConfigError, build_agent, parse_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nexus-service",
        description="Run the nexus-agents HTTP service (session-scoped chat + SSE).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the environment, print the resolved config and exit",
    )
    args = parser.parse_args(argv)

    try:
        config = parse_env()
        agent = build_agent(config)
    except (EnvConfigError, MissingCredentialsError) as exc:
        # 拼错一个变量名就启动失败，比带着错误配置跑起来诚实得多
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(config.describe())
        return 0

    try:
        import uvicorn
    except ModuleNotFoundError:  # pragma: no cover - 只在没装 api extra 时触发
        print('uvicorn is not installed: pip install "nexus-agents[api]"', file=sys.stderr)
        return 2

    app = create_app(agent, settings=config.service)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        # 收到 SIGTERM 后先让在途请求收尾：SSE 多开一秒，总比客户端看到
        # 一个没有任何结尾帧的截断流要好
        timeout_graceful_shutdown=config.graceful_timeout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 只由解释器直接执行时走到
    raise SystemExit(main())
