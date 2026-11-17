# 刻意不写 `# syntax=docker/dockerfile:1`
# =====================================
# 这条指令会让构建先去 docker.io 拉一个 frontend 镜像才能解析 Dockerfile。
# 本仓库不需要任何新语法（没有 heredoc、没有 RUN --mount），却因此多了一个
# 「构建必须先连上 docker.io」的失败点 —— 在代理后面、或完全离线时直接报
# `encountered unknown type text/html`。用内置前端就够了。
#
# 两个阶段，各自只解决一个问题
# ============================
# builder  stage：把源码和依赖编译成 wheel。它需要 pip 的构建流程，
#                但**不需要**活到运行时。
# runtime  stage：只装 wheel，不带 PEP 517 构建链、不带源码目录、不带缓存。
#
# 收益不只是「镜像更小」（实测 1.4GB → 200MB 量级）：运行阶段里连
# setuptools 都没有，也就少了一整类「构建期依赖被当成运行期依赖」的隐患，
# 更少的东西能被打包进 CVE 报告。
#
# 为什么先拷 pyproject.toml 再拷 src
# ----------------------------------
# Docker 的层缓存以指令为单位。依赖声明（pyproject）的改动频率远低于业务
# 代码；先单独拷贝它并把依赖装完，改一行 Python 只会让最后一层失效，
# 不用重装 pydantic/fastapi。这一条能把日常构建从分钟级降到秒级。

# ---------------------------------------------------------------- builder
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

# 只拷「构建 wheel 需要的最小集合」：README 是 pyproject 里 readme 字段
# 指向的文件，缺了它 setuptools 会直接失败。
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# 一次把所有 wheel（本项目 + 全部第三方依赖）落到 /wheels，
# 后面运行阶段就能 --no-index 离线安装，不再访问网络。
RUN python -m pip wheel --wheel-dir /wheels ".[api]"

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

# PYTHONUNBUFFERED：容器里 stdout 是管道，不关缓冲会丢掉崩溃前最后几行日志
# PYTHONDONTWRITEBYTECODE：镜像只读，写 .pyc 既没用又会在只读根文件系统上告警
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    NEXUS_HOST=0.0.0.0 \
    NEXUS_PORT=8000

# 非 root 运行：容器逃逸的绝大部分利用链第一步就是拿 root。
# 固定 uid/gid 而不是让系统分配，K8s 的 securityContext 才能对得上。
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels "nexus-agents[api]" \
    && rm -rf /wheels

USER app
WORKDIR /home/app

EXPOSE 8000

# 探针用标准库而不是装 curl：镜像里每多一个二进制就多一份漏洞面，
# 而 urllib 已经随 Python 一起进来了。非 200 会抛异常 → 退出码非零 → unhealthy。
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('NEXUS_PORT','8000'), timeout=2)"

# 用模块入口而不是 nexus-service 脚本：不依赖 PATH，排错时也少一层间接。
CMD ["python", "-m", "nexus_agents.service"]
