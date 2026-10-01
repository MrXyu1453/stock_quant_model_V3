"""用户认证模块：注册、登录、会话管理、当前用户上下文"""
import logging

from flask import session, has_request_context

from .security import hash_password, verify_password
from . import database_manager as db


def register_user(username, password):
    """注册新用户，返回 (ok, msg, user_id)"""
    username = (username or '').strip()
    if not username:
        return False, '用户名不能为空', None
    if len(username) < 2:
        return False, '用户名至少 2 个字符', None
    if not password or len(password) < 6:
        return False, '密码至少 6 位', None
    if db.username_exists(username):
        return False, '用户名已存在', None
    uid = db.insert_user(username, hash_password(password))
    if uid is None:
        return False, '注册失败，请稍后重试', None
    return True, '注册成功', uid


def authenticate(username, password):
    """登录校验，返回 (ok, msg, user_id)"""
    username = (username or '').strip()
    row = db.get_user_by_username(username)
    if not row:
        return False, '用户名不存在', None
    uid, pwd_hash = row[0], row[1]
    if not verify_password(password, pwd_hash):
        return False, '密码错误', None
    return True, '登录成功', uid


def change_password(uid, old_password, new_password):
    """修改密码，返回 (ok, msg)"""
    if not new_password or len(new_password) < 6:
        return False, '新密码至少 6 位'
    row = db.get_user_by_id(uid)
    if not row:
        return False, '用户不存在'
    stored = db.get_user_by_username(row['username'])
    if not stored or not verify_password(old_password, stored[1]):
        return False, '原密码错误'
    db.update_user_password(uid, hash_password(new_password))
    return True, '密码修改成功'


def login_session(uid, username):
    """写入登录会话"""
    session['user_id'] = uid
    session['username'] = username
    session.permanent = True


def logout_session():
    """清空登录会话"""
    session.clear()


def get_current_user():
    """获取当前登录用户 {'id':.., 'username':..}，未登录返回 None"""
    try:
        if has_request_context():
            uid = session.get('user_id')
            username = session.get('username')
            if uid is not None and username is not None:
                return {'id': uid, 'username': username}
    except Exception as e:
        logging.error(f"获取当前用户时出错: {e}")
    return None


def get_current_username():
    user = get_current_user()
    return user['username'] if user else None
