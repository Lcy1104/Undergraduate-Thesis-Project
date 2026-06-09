# login.py
import streamlit as st
import time
import sys
from streamlit_js import st_js
import streamlit.components.v1 as components
from urllib.parse import unquote
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db import *
import time
from db import _get_cached_db_client, _get_cached_encoder
# 设置页面配置（与 db.py 保持一致）
st.set_page_config(page_title="知识库问答系统", page_icon="📚", layout="wide")
st.markdown("""
<style>
    /* 隐藏侧边栏中的多页面导航链接（包括主应用和 pages 下的页面） */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    /* 可选：调整侧边栏顶部间距，避免空白过多 */
    section[data-testid="stSidebar"] > div:first-child {
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)


def auth_page():
    # 如果已登录，直接跳转
    if st.session_state.get("is_logged_in", False):
        st.switch_page("db.py")
        return

    # 处理 URL 中的 token（自动登录）
    token = st.query_params.get("token")
    if token:
        token = unquote(token)
        try:
            decrypted = sm4_decrypt(token)
            if "|" in decrypted:
                username, expire_ts = decrypted.split("|", 1)
                if time.time() < int(expire_ts):
                    db_client = _get_cached_db_client()
                    if db_client:
                        user_data = db_client.get_user_by_username(username)
                        if user_data:
                            st.session_state.is_logged_in = True
                            st.session_state.current_user = username
                            st.session_state.user_permissions = user_data.get("permissions", Permissions.DEFAULT_USER)
                            st.session_state.db_client = db_client
                            st.session_state.encoder = _get_cached_encoder()
                            st.session_state.db_client_ready = True
                            #st.query_params.clear()
                            # 设置 URL 参数并刷新
                            st.query_params["token"] = token
                            st.rerun()
        except Exception as e:
            print(f"[login.py] 自动登录异常: {e}")
            js_code = f"""
                        window.location.href = "db.py?token=" + encodeURIComponent("{token}");
                        return "redirected";
                        """
            st_js(js_code, key="auto_login_redirect")
            st.stop()
        st.query_params.pop("token", None)

    # ========== 显示登录表单 ==========
    print("[login.py] 进入手动登录流程")

    # 初始化数据库连接
    db_client = _get_cached_db_client()
    encoder = _get_cached_encoder()

    if db_client is None or encoder is None:
        st.error("❌ 数据库配置无效，请返回主页面重新初始化")
        if st.button("返回主页面"):
            st.switch_page("db.py")
        return

    st.session_state.db_client = db_client
    st.session_state.encoder = encoder
    st.session_state.db_client_ready = True
    st.session_state.init_ok = False

    # ========== 1. 初始化验证码 ==========
    init_captcha_session()

    # ========== 2. 初始化session_state（兜底） ==========
    init_auth_state = {
        "auth_panel": "login",
        "model_type": "双模型融合",
        "use_cuda": False,
        "login_user": "", "login_pwd": "", "login_captcha": "",
        "reg_user": "", "reg_pwd": "", "reg_confirm": "", "reg_captcha": "",
        "clear_login_input": False, "clear_reg_input": False,
        "is_logged_in": False, "init_ok": False, "is_first_init": False,
        "reset_verified_user": None  # 新增：修改密码验证的用户名
    }
    for key, val in init_auth_state.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ========== 3. 处理清空输入标记 ==========
    if st.session_state.clear_login_input:
        st.session_state.login_user = ""
        st.session_state.login_pwd = ""
        st.session_state.login_captcha = ""
        st.session_state.clear_login_input = False
        st.rerun()
    if st.session_state.clear_reg_input:
        st.session_state.reg_user = ""
        st.session_state.reg_pwd = ""
        st.session_state.reg_confirm = ""
        st.session_state.reg_captcha = ""
        st.session_state.clear_reg_input = False
        st.rerun()

    # ========== 4. 登录面板标题+分割线 ==========
    st.subheader("📚 知识库问答系统")
    st.divider()

    # ========== 5. 根据当前面板渲染对应的表单（内联函数定义） ==========
    current_panel = st.session_state.auth_panel

    def render_login_panel():
        """渲染登录面板"""
        st.caption("请输入账号密码登录")

        # 输入框
        st.text_input("用户名", placeholder="请输入你的账号", key="login_user_input")
        st.text_input("密码", placeholder="请输入你的密码", type="password", key="login_pwd_input")

        # 验证码
        st.divider()
        col_img, col_input, col_refresh = st.columns([1.5, 1, 0.5])
        with col_img:
            img = st.session_state.get("captcha_image") or generate_captcha_image()
            st.image(img, width=140)
        with col_input:
            st.text_input(label="验证码", placeholder="输入4位字符",
                          key="login_captcha_input", label_visibility="collapsed")
        with col_refresh:
            if st.button("🔄", key="login_refresh_btn", use_container_width=True):
                refresh_captcha()
                st.rerun()

        # 按钮
        col_switch, col_reset_pwd, col_login, col_reset = st.columns(4)

        with col_switch:
            if st.button("注册账号", type="secondary", key="login_to_register_btn"):
                st.session_state.auth_panel = "register"
                refresh_captcha()
                st.rerun()

        with col_reset_pwd:
            if st.button("修改密码", type="secondary", key="login_to_reset_btn"):
                st.session_state.auth_panel = "reset_password_verify"
                refresh_captcha()
                st.rerun()

        with col_login:
            if st.button("登录", type="primary", key="login_submit_btn"):
                handle_login()

        with col_reset:
            user = st.session_state.get("login_user_input", "").strip()
            pwd = st.session_state.get("login_pwd_input", "").strip()
            captcha = st.session_state.get("login_captcha_input", "").strip()
            disabled = not (user or pwd or captcha)
            if st.button("重置", key="login_reset_btn", disabled=disabled):
                st.session_state.login_user_input = ""
                st.session_state.login_pwd_input = ""
                st.session_state.login_captcha_input = ""
                st.rerun()

    def render_register_panel():
        """渲染注册面板"""
        st.caption("请填写信息完成注册")

        # 输入框
        st.text_input("新用户名", key="reg_user_input")
        st.text_input("新密码（≥6位）", type="password", key="reg_pwd_input")
        st.text_input("确认新密码", type="password", key="reg_confirm_input")

        # 验证码
        st.divider()
        col_img, col_input, col_refresh = st.columns([1.5, 1, 0.5])
        with col_img:
            img = st.session_state.get("captcha_image") or generate_captcha_image()
            st.image(img, width=140)
        with col_input:
            st.text_input(label="验证码", placeholder="输入4位字符",
                          key="reg_captcha_input", label_visibility="collapsed")
        with col_refresh:
            if st.button("🔄", key="reg_refresh_btn", use_container_width=True):
                refresh_captcha()
                st.rerun()

        # 按钮
        col_submit, col_cancel, col_reset = st.columns(3)

        with col_submit:
            if st.button("确认注册", type="primary", key="reg_submit_btn"):
                handle_register()

        with col_cancel:
            if st.button("取消", type="secondary", key="reg_cancel_btn"):
                st.session_state.auth_panel = "login"
                st.rerun()

        with col_reset:
            user = st.session_state.get("reg_user_input", "").strip()
            pwd = st.session_state.get("reg_pwd_input", "").strip()
            confirm = st.session_state.get("reg_confirm_input", "").strip()
            captcha = st.session_state.get("reg_captcha_input", "").strip()
            disabled = not (user or pwd or confirm or captcha)
            if st.button("重置", key="reg_reset_btn", disabled=disabled):
                st.session_state.reg_user_input = ""
                st.session_state.reg_pwd_input = ""
                st.session_state.reg_confirm_input = ""
                st.session_state.reg_captcha_input = ""
                st.rerun()

    def render_reset_verify_panel():
        """渲染重置密码-验证用户面板"""
        st.caption("🔐 修改密码 - 请输入用户名验证身份")

        # 输入框
        st.text_input("用户名", key="reset_user_input")

        # 按钮
        col_back, col_next = st.columns(2)

        with col_back:
            if st.button("返回登录", type="secondary", key="reset_back_to_login_btn"):
                st.session_state.auth_panel = "login"
                st.rerun()

        with col_next:
            if st.button("下一步", type="primary", key="reset_next_btn"):
                handle_reset_verify()

    def render_reset_set_panel():
        """渲染重置密码-设置新密码面板"""
        verified_user = st.session_state.get("reset_verified_user")
        if not verified_user:
            st.error("❌ 请先完成用户名验证！")
            st.session_state.auth_panel = "reset_password_verify"
            st.rerun()


        st.caption(f"🔐 设置新密码（用户：{verified_user}）")

        # 输入框
        st.text_input("新密码（≥6位）", type="password", key="reset_new_pwd_input")
        st.text_input("确认新密码", type="password", key="reset_confirm_pwd_input")

        # 验证码
        col_img, col_input, col_refresh = st.columns([1.5, 1, 0.5])
        with col_img:
            img = st.session_state.get("captcha_image") or generate_captcha_image()
            st.image(img, width=140)
        with col_input:
            st.text_input(label="验证码", placeholder="4位字符",
                          key="reset_captcha_input", label_visibility="collapsed")
        with col_refresh:
            if st.button("🔄", key="reset_refresh_btn", use_container_width=True):
                refresh_captcha()
                st.rerun()

        # 按钮
        col_back, col_submit = st.columns(2)

        with col_back:
            if st.button("返回上一步", type="secondary", key="reset_back_step_btn"):
                st.session_state.auth_panel = "reset_password_verify"
                st.rerun()

        with col_submit:
            if st.button("确认修改", type="primary", key="reset_submit_btn"):
                handle_reset_set()

    # 在 auth_page 函数中的 handle_login 函数里，改进验证码处理

    def handle_login():
        """处理登录逻辑"""
        user = st.session_state.get("login_user_input", "").strip()
        pwd = st.session_state.get("login_pwd_input", "").strip()
        captcha = st.session_state.get("login_captcha_input", "").strip()
        captcha_text = st.session_state.get("captcha_text", "").strip()

        if not user or not pwd or not captcha:
            st.error("❌ 账号/密码/验证码不能为空！")
            return

        # 🔧 修复：更详细的验证码验证调试
        user_captcha = captcha.strip().upper()
        correct_captcha = captcha_text.strip().upper()

        # 调试信息（生产环境可注释掉）
        # st.write(f"DEBUG: 用户输入: '{user_captcha}'")
        # st.write(f"DEBUG: 正确验证码: '{correct_captcha}'")
        # st.write(f"DEBUG: 是否相等: {user_captcha == correct_captcha}")

        if user_captcha != correct_captcha:
            st.error("❌ 验证码错误！")
            # 增加尝试次数
            if 'captcha_attempts' not in st.session_state:
                st.session_state.captcha_attempts = 0
            st.session_state.captcha_attempts += 1

            if st.session_state.captcha_attempts >= 3:
                st.error("🚫 验证码尝试次数过多，请刷新页面！")
                st.session_state.captcha_attempts = 0
                time.sleep(2)
                st.rerun()

            refresh_captcha()
            st.rerun()

        # 重置验证码尝试次数
        if 'captcha_attempts' in st.session_state:
            st.session_state.captcha_attempts = 0

        db_client = st.session_state.db_client
        if db_client is None:
            st.error("❌ 数据库未初始化")
            return

        encrypt_pwd = sm4_encrypt(pwd)
        user_data = db_client.get_user_by_username(user)

        if user_data and user_data["password"] == encrypt_pwd:
            # 检查用户是否有登录权限
            user_permissions = user_data.get("permissions", Permissions.DEFAULT_USER)
            if not Permissions.has_permission(user_permissions, Permissions.LOGIN):
                st.error("❌ 该用户没有登录权限！")
                refresh_captcha()
                st.rerun()
            # 生成 token
            token_data = f"{user}|{int(time.time()) + 86400 * 7}"
            token = sm4_encrypt(token_data)
            # 保存用户信息和权限到session_state
            st.session_state.is_logged_in = True
            st.session_state.current_user = user
            st.session_state.user_permissions = user_permissions
            st.session_state.db_client = db_client
            st.session_state.encoder = _get_cached_encoder()
            st.session_state.db_client_ready = True

            # 🔧 记录登录日志
            log_sensitive_operation(
                "用户登录",
                user,
                user,
                f"权限: {user_permissions}"
            )

            # 使用客户端 meta refresh 跳转（确保 URL 显示 token）
            st.markdown(f"""
                <meta http-equiv="refresh" content="0; url=/?token={token}">
            """, unsafe_allow_html=True)
            try:
                if st.session_state.db_client:
                    st.session_state.db_client.conn.close()
                    print("[login.py] 数据库连接已关闭")
            except:
                pass
            st.stop()
        else:
            st.error("❌ 账号/密码错误或未注册")
            refresh_captcha()
            st.rerun()

    def handle_register():
        """处理注册逻辑"""
        user = st.session_state.get("reg_user_input", "").strip()
        pwd = st.session_state.get("reg_pwd_input", "").strip()
        confirm = st.session_state.get("reg_confirm_input", "").strip()
        captcha = st.session_state.get("reg_captcha_input", "").strip()
        captcha_text = st.session_state.get("captcha_text", "").strip()

        if not user or not pwd or not confirm or not captcha:
            st.error("❌ 所有字段不能为空！")
            return

        # 🔧 修复：禁止注册admin用户
        if user.lower() == "admin":
            st.error("❌ 不能注册为管理员用户！")
            refresh_captcha()
            st.rerun()

        if len(pwd) < 6:
            st.error("❌ 密码长度≥6位！")
            return

        if pwd != confirm:
            st.error("❌ 两次密码不一致！")
            return

        if captcha.upper() != captcha_text.upper():
            st.error("❌ 验证码错误！")
            refresh_captcha()
            st.rerun()

        db_client = st.session_state.db_client
        res = db_client.register_user(user, pwd)

        if res == "success":
            st.success("✅ 注册成功！自动跳转登录")
            st.session_state.auth_panel = "login"
            refresh_captcha()
            st.rerun()
        elif res == "exists":
            st.error("❌ 用户名已存在！")
            refresh_captcha()
            st.rerun()
        else:
            st.error("❌ 注册失败，请重试！")
            refresh_captcha()
            st.rerun()

    def handle_reset_verify():
        """处理重置密码验证"""
        user = st.session_state.get("reset_user_input", "").strip()

        if not user:
            st.error("❌ 用户名不能为空！")
            st.rerun()

        if not st.session_state.db_client.check_user_exists(user):
            st.error("❌ 该用户名不存在！")
            st.rerun()

        st.session_state.reset_verified_user = user
        st.session_state.auth_panel = "reset_password_set"
        st.rerun()

    def handle_reset_set():
        """处理重置密码设置"""
        user = st.session_state.get("reset_verified_user")
        new_pwd = st.session_state.get("reset_new_pwd_input", "").strip()
        confirm_pwd = st.session_state.get("reset_confirm_pwd_input", "").strip()
        captcha = st.session_state.get("reset_captcha_input", "").strip()
        captcha_text = st.session_state.get("captcha_text", "").strip()

        if not new_pwd or not confirm_pwd or not captcha:
            st.error("❌ 所有项都不能为空！")
            st.rerun()

        if new_pwd != confirm_pwd:
            st.error("❌ 两次密码不一致！")
            st.rerun()

        if len(new_pwd) < 6:
            st.error("❌ 密码长度≥6位！")
            st.rerun()

            # 🔧 修复验证码比较逻辑
        if captcha.upper() != captcha_text.upper():  # 修改为统一大写比较
            st.error("❌ 验证码错误！")
            refresh_captcha()
            st.rerun()

        if st.session_state.db_client.update_user_password(user, new_pwd):
            st.success("✅ 密码修改成功！请用新密码登录")
            st.session_state.auth_panel = "login"
            st.session_state.pop("reset_verified_user", None)
            refresh_captcha()
            time.sleep(1.5)
            st.rerun()
        else:
            st.error("❌ 修改失败，请重试！")
            refresh_captcha()
            st.rerun()

    # ========== 6. 根据当前面板调用对应的渲染函数 ==========
    if current_panel == "login":
        render_login_panel()
    elif current_panel == "register":
        render_register_panel()
    elif current_panel == "reset_password_verify":
        render_reset_verify_panel()
    elif current_panel == "reset_password_set":
        render_reset_set_panel()


# ====================== 登录页面函数结束 ======================
if __name__ == "__main__":
    auth_page()