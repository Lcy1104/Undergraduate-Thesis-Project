import redis
from redis.exceptions import ConnectionError, TimeoutError
import sys


def test_redis_detailed():
    print("开始测试Redis连接...")
    print(f"目标: localhost:6379")
    print("-" * 50)

    try:
        # 创建连接池
        pool = redis.ConnectionPool(
            host='localhost',
            port=6379,
            db=0,
            password='Password123@redis',  # ← 这里
            max_connections=10,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )

        r = redis.Redis(connection_pool=pool)

        # 测试1: PING命令
        print("测试1: 发送PING命令...")
        ping_result = r.ping()
        print(f"  PING响应: {ping_result}")

        # 测试2: 设置和获取键值
        print("\n测试2: 设置和获取键值...")
        test_key = "test:python:connection"
        test_value = "Hello Redis from Python!"

        r.set(test_key, test_value, ex=10)  # 设置10秒过期
        retrieved_value = r.get(test_key)

        if retrieved_value and retrieved_value.decode('utf-8') == test_value:
            print(f"  ✓ 键值存储测试成功")
            print(f"  设置的值: {test_value}")
            print(f"  获取的值: {retrieved_value.decode('utf-8')}")
        else:
            print("  ✗ 键值存储测试失败")

        # 测试3: 获取Redis信息
        print("\n测试3: 获取Redis服务器信息...")
        info = r.info()
        print(f"  Redis版本: {info.get('redis_version', 'N/A')}")
        print(f"  运行模式: {info.get('redis_mode', 'N/A')}")
        print(f"  内存使用: {info.get('used_memory_human', 'N/A')}")
        print(f"  连接客户端数: {info.get('connected_clients', 'N/A')}")

        # 测试4: 检查是否支持某些命令
        print("\n测试4: 检查Redis功能...")
        try:
            r.time()  # Redis 2.6+ 支持
            print("  ✓ 支持TIME命令 (Redis 2.6+)")
        except:
            print("  ✗ 不支持TIME命令")

        print("\n" + "=" * 50)
        print("🎉 所有测试通过！Redis连接正常。")

        return True

    except ConnectionError as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n可能的原因:")
        print("1. Redis服务未启动")
        print("2. 防火墙阻止了连接")
        print("3. Redis配置绑定了其他IP")
        print("\n建议:")
        print(f"  确保Redis正在运行: redis-server.exe")
        print(f"  检查端口是否开放: netstat -an | findstr 6379")

    except TimeoutError as e:
        print(f"\n❌ 连接超时: {e}")

    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

    return False


if __name__ == "__main__":
    success = test_redis_detailed()
    sys.exit(0 if success else 1)