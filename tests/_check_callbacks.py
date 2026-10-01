"""开发辅助（进程内）：渲染各页面布局，校验 callback 引用的组件 ID 都存在"""
import sys

from app import dash_app as m
import app.factor_page as fp
import app.market_overview as mo
import app.sector_fund_strength as sfs
import app.trend_analysis as ta
import app.stock_game as sg
import app.trading_journal as tj


def collect_ids(component, ids):
    if hasattr(component, 'id') and component.id is not None:
        ids.add(str(component.id))
    children = getattr(component, 'children', None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            collect_ids(child, ids)


def main():
    pages = {
        '/': m.analysis_layout,
        '/stock-management': m.stock_management_layout,
        '/factor-training': fp.factor_page_layout,
        '/trend-analysis': ta.trend_analysis_layout,
        '/stock-game': sg.stock_game_layout,
        '/market-overview': mo.market_overview_layout,
        '/sector-fund-strength': sfs.sector_fund_strength_layout,
        '/trading-journal': tj.trading_journal_layout,
    }

    all_ids = set()
    for name, fn in pages.items():
        ids = set()
        collect_ids(fn(), ids)
        print(f'{name:24s} ids={len(ids):4d}')
        all_ids |= ids

    # app.layout 是运行时真正的完整树（含登录页/header/弹窗）
    layout_ids = set()
    collect_ids(m.app.layout, layout_ids)
    all_ids |= layout_ids
    print(f'\n布局并集 ids={len(all_ids)}')

    cb_refs = set()
    for key, spec in m.app.callback_map.items():
        for out in key.split('..'):
            cb_refs.add(out.split('.')[0].strip())
        for inp in spec['inputs']:
            cb_refs.add(inp['id'])
        for st in spec.get('state', []):
            cb_refs.add(st['id'])

    # 模式匹配回调的 ID 形如 {"index":..., "type":...}，由其它 callback 动态生成，跳过静态检查
    dynamic_refs = {r for r in cb_refs if r.startswith('{')}
    missing = cb_refs - all_ids - dynamic_refs
    print(f'callback 数={len(m.app.callback_map)}  静态引用组件数={len(cb_refs - dynamic_refs)}  动态模式引用={len(dynamic_refs)}')
    if missing:
        print('\n[!] callback 引用但所有布局都不存在的 ID:')
        for x in sorted(missing):
            print('   -', x)
        return 1
    print('\n[OK] 所有 callback 引用的静态组件 ID 均存在于布局中')
    return 0


if __name__ == '__main__':
    sys.exit(main())
