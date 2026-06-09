#!/usr/bin/env python3
"""
专门修复admin用户问题的脚本 - 修改版
"""

import sys
import os
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.append(os.path.dirname(__file__))


def sm4_encrypt(text):
    """简单的SM4加密函数（仅用于修复脚本）"""
    import hashlib
    # 使用简单的哈希代替SM4，用于修复脚本
    return hashlib.md5(text.encode()).hexdigest()


def fix_admin_users():
    """修复admin用户问题"""

    # 数据库连接配置（根据您的实际配置修改）
    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "root",  # 您的密码
        "dbname": "kb_db"
    }

    try:
        # 连接到数据库
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print("🔧 开始修复admin用户问题...")

        # 加密的admin用户名
        encrypt_admin_user = sm4_encrypt("admin")

        # 1. 查找所有admin用户
        cursor.execute(f"""
            SELECT id, username, permissions, create_time 
            FROM system_users 
            WHERE username = '{encrypt_admin_user}'
            ORDER BY create_time ASC;
        """)
        admin_users = cursor.fetchall()

        print(f"发现 {len(admin_users)} 个admin用户")

        if len(admin_users) > 1:
            print("⚠️ 存在多个admin用户，正在修复...")

            # 保留最早创建的admin用户
            keep_admin = admin_users[0]
            delete_admins = admin_users[1:]

            print(f"保留ID为 {keep_admin['id']} 的admin用户（创建于: {keep_admin['create_time']}）")

            # 删除其他admin用户
            delete_count = 0
            for admin in delete_admins:
                cursor.execute("DELETE FROM system_users WHERE id = %s;", (admin['id'],))
                delete_count += 1
                print(f"  删除ID为 {admin['id']} 的admin用户")

            print(f"✅ 删除了 {delete_count} 个重复的admin用户")

        # 2. 确保admin用户权限正确（255 = 所有权限）
        cursor.execute(f"""
            UPDATE system_users 
            SET permissions = 255 
            WHERE username = '{encrypt_admin_user}';
        """)

        print("✅ 确保admin用户权限正确（权限: 255）")

        # 提交更改
        conn.commit()

        # 3. 验证修复结果
        cursor.execute(f"""
            SELECT COUNT(*) as count FROM system_users 
            WHERE username = '{encrypt_admin_user}';
        """)
        final_admin_count = cursor.fetchone()['count']

        cursor.execute(f"""
            SELECT username, permissions FROM system_users 
            WHERE permissions = 255;
        """)
        all_admins = cursor.fetchall()

        print("\n📊 修复完成后的状态：")
        print(f"  - Admin用户数量: {final_admin_count}")
        print(f"  - 所有管理员权限用户: {len(all_admins)} 个")

        for admin in all_admins:
            status = "（系统管理员）" if admin['username'] == encrypt_admin_user else "（其他管理员）"
            print(f"    - {admin['username'][:10]}... 权限: {admin['permissions']} {status}")

        print("\n🎉 Admin用户问题修复完成！")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()


def reset_all_users_permissions():
    """重置所有用户权限到初始值（不删除用户）"""

    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "root",  # 您的密码
        "dbname": "kb_db"
    }

    response = input("⚠️ 警告：这将重置所有用户的权限到初始值。确定继续？(输入'YES'继续): ")
    if response != "YES":
        print("操作已取消")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print("🔄 正在重置所有用户权限...")

        # 加密的admin用户名
        encrypt_admin_user = sm4_encrypt("admin")

        # 普通用户初始权限（根据新的权限逻辑：登录 + 智能问答 + 图片识别问答 = 97）
        # LOGIN = 1, USE_QA = 32, USE_IMAGE_QA = 64，总和 = 97
        normal_user_permissions = 97  # 只有问答功能，看不到知识库

        # 管理员权限
        admin_permissions = 255  # 所有权限

        # 1. 重置所有非admin用户的权限为97（普通用户权限）
        cursor.execute(f"""
            UPDATE system_users 
            SET permissions = {normal_user_permissions} 
            WHERE username != '{encrypt_admin_user}';
        """)

        normal_users_updated = cursor.rowcount
        print(f"✅ 重置了 {normal_users_updated} 个普通用户的权限为 {normal_user_permissions}")

        # 2. 确保admin用户权限为255（所有权限）
        cursor.execute(f"""
            SELECT COUNT(*) as count FROM system_users 
            WHERE username = '{encrypt_admin_user}';
        """)
        admin_exists = cursor.fetchone()['count'] > 0

        if admin_exists:
            cursor.execute(f"""
                UPDATE system_users 
                SET permissions = {admin_permissions} 
                WHERE username = '{encrypt_admin_user}';
            """)
            print(f"✅ 重置admin用户权限为 {admin_permissions}")
        else:
            # 如果admin不存在，创建它
            encrypt_admin_pwd = sm4_encrypt("admin")
            cursor.execute(f"""
                INSERT INTO system_users (username, password, permissions)
                VALUES ('{encrypt_admin_user}', '{encrypt_admin_pwd}', {admin_permissions});
            """)
            print("✅ 创建admin用户并设置权限为 255")

        conn.commit()

        # 3. 验证结果
        cursor.execute("SELECT COUNT(*) as count FROM system_users;")
        total_users = cursor.fetchone()['count']

        cursor.execute(f"""
            SELECT username, permissions FROM system_users 
            WHERE username = '{encrypt_admin_user}';
        """)
        admin_user = cursor.fetchone()

        cursor.execute(f"""
            SELECT username, permissions FROM system_users 
            WHERE username != '{encrypt_admin_user}';
        """)
        normal_users = cursor.fetchall()

        print(f"\n📊 重置完成后的状态：")
        print(f"  - 总用户数: {total_users}")
        if admin_user:
            print(f"  - Admin用户权限: {admin_user['permissions']} (所有权限)")
        print(f"  - 普通用户数量: {len(normal_users)}")

        if normal_users:
            print("  普通用户列表：")
            for user in normal_users[:10]:  # 显示前10个
                print(f"    - {user['username'][:10]}... 权限: {user['permissions']}")
            if len(normal_users) > 10:
                print(f"    ... 还有 {len(normal_users) - 10} 个用户")

        cursor.close()
        conn.close()

        print("\n🔐 权限重置完成！")
        print("   - Admin用户拥有所有权限 (255)")
        print("   - 普通用户只有登录和问答功能 (97)")
        print("   - 需要管理员授权，普通用户才能看到知识库管理")

    except Exception as e:
        print(f"❌ 重置失败: {e}")
        import traceback
        traceback.print_exc()


def delete_all_users_except_admin():
    """删除所有非admin用户（危险操作，仅用于紧急情况）"""

    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "root",  # 您的密码
        "dbname": "kb_db"
    }

    response = input("⚠️ 危险操作：这将删除所有非admin用户。确定继续？(输入'YES'继续): ")
    if response != "YES":
        print("操作已取消")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print("🗑️ 正在删除所有非admin用户...")

        # 加密的admin用户名
        encrypt_admin_user = sm4_encrypt("admin")
        encrypt_admin_pwd = sm4_encrypt("admin")

        # 1. 删除所有非admin用户
        cursor.execute(f"""
            DELETE FROM system_users 
            WHERE username != '{encrypt_admin_user}';
        """)

        deleted_count = cursor.rowcount
        print(f"✅ 删除了 {deleted_count} 个非admin用户")

        # 2. 确保admin用户存在且权限正确
        cursor.execute(f"""
            SELECT COUNT(*) as count FROM system_users 
            WHERE username = '{encrypt_admin_user}';
        """)
        admin_exists = cursor.fetchone()['count'] > 0

        if not admin_exists:
            cursor.execute(f"""
                INSERT INTO system_users (username, password, permissions)
                VALUES ('{encrypt_admin_user}', '{encrypt_admin_pwd}', 255);
            """)
            print("✅ 重新创建admin用户")
        else:
            cursor.execute(f"""
                UPDATE system_users 
                SET permissions = 255 
                WHERE username = '{encrypt_admin_user}';
            """)
            print("✅ 重置admin用户权限")

        conn.commit()

        # 3. 验证结果
        cursor.execute("SELECT COUNT(*) as count FROM system_users;")
        total_users = cursor.fetchone()['count']

        print(f"\n📊 删除完成后的状态：")
        print(f"  - 总用户数: {total_users}")
        print(f"  - 所有非admin用户已被删除")

        cursor.close()
        conn.close()

        print("\n🔐 现在您可以使用 admin/admin 登录")

    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("用户权限管理工具")
    print("=" * 60)
    print("1. 修复重复的admin用户")
    print("2. 重置所有用户权限（不删除用户）")
    print("3. 删除所有非admin用户（危险！仅保留admin）")
    print("4. 退出")

    choice = input("\n请选择操作 (1-4): ").strip()

    if choice == "1":
        fix_admin_users()
    elif choice == "2":
        reset_all_users_permissions()
    elif choice == "3":
        delete_all_users_except_admin()
    elif choice == "4":
        print("退出")
    else:
        print("无效选择")

