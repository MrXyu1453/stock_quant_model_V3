#!/usr/bin/env python

import os
import sys

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入并运行 Dash 应用
from app.dash_app import app

# 启动后台自选股信号扫描器
# debug 重载模式下只有真正提供服务的子进程（WERKZEUG_RUN_MAIN=true）启动，父进程不扫，避免双份请求
DEBUG = True  # 与下方 app.run(debug=DEBUG) 保持一致
from app.signal_scanner import start_background_scanner

if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not DEBUG:
    start_background_scanner()

if __name__ == '__main__':
    # 确保在 app 包上下文中运行
    app.run(debug=DEBUG)
