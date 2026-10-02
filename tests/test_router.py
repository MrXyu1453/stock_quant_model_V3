"""路由回调返回值对齐测试：防止 style/children 输出错位（React 渲染报错）"""
from dash import html

from app import dash_app as dd


def test_router_unauthorized_alignment():
    """未登录: 11 个输出按序对齐，page-header.children 必须是组件而非 style 字典"""
    result = dd.router('/', None)
    assert len(result) == 11
    assert result[0] == {'display': 'block'}     # login-page.style
    assert result[1] == {'display': 'none'}      # app-content.style
    assert isinstance(result[2], html.Header), \
        'page-header.children 若收到 style 字典会触发 React 报错'
    for style in result[3:]:
        assert isinstance(style, dict) and set(style.keys()) == {'display'}


def test_router_authorized_alignment(monkeypatch):
    monkeypatch.setattr(dd.auth, 'get_current_user',
                        lambda: {'id': 1, 'username': 'tester'})
    result = dd.router('/stock-game', {'user_id': 1})
    assert len(result) == 11
    assert result[0] == {'display': 'none'}      # login-page 隐藏
    assert result[1] == {'display': 'block'}     # app-content 显示
    assert isinstance(result[2], html.Header)
    assert result[7] == {'display': 'block'}, '/stock-game 应显示 game-page'
    assert result[3] == {'display': 'none'}      # 分析页隐藏
