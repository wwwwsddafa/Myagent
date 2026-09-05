"""日志与监控工具"""

import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s -[%(name)s]- %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_assistant")


# 装饰器:有点像java中的注解的用法
def log_node_execution(node_name: str):
    """装饰器:记录各个节点执行时间和结果"""

    def decorator(func):  # func代表被装饰的函数，(java的aop理解的话联接点 : 环绕通知 )
        def wrapper(state):  # state代表节点的输入状态
            logger.info(f"[{node_name}] 开始执行")
            start = time.time()

            result = func(state)  # 执行节点函数

            elapsed = time.time() - start
            logger.info(f"[{node_name}] 完成, 耗时 {elapsed:.2f}s")
            # 记录结果摘要
            if isinstance(result, dict):  # 说明当前执行的节点的返回值是一个字典
                for key, value in result.items():
                    if isinstance(value, str):
                        logger.info(f"[{node_name}] 输出 {key}: {len(value)} 字符")
                    elif isinstance(value, list):
                        logger.info(f"[{node_name}] 输出 {key}: {len(value)} 项")
            else:  # 这个节点是路由节点, 返回值是一个字符串
                logger.info(f"[{node_name}] 输出: {result}")
            return result

        return wrapper

    return decorator

"""
log_node_execution(node_name: str):接收一个节点函数的名字，返回一个装饰器。
decorator(func):它的唯一工作就是接收你的原始节点函数 (func)，然后制造并返回一个新的函数 (wrapper)。
wrapper(state):它是最终被调用的函数。当你在代码里调用 classify_question(state) 时，实际上运行的是 wrapper。
wrapper是怎么干活的：
  拦截：先截获你传入的参数 state。
  加料：执行额外的逻辑（记日志、掐表计时）。
  放行：调用被锁在闭包里的原始函数 func(state)。
  收尾：拿到结果，再记一次日志，最后把结果交给你。

"""



def log_tool_execution(node_name: str):
    """装饰器:记录各个节点执行时间和结果"""

    def decorator(func):  # func代表被装饰的函数，(java的aop理解的话联接点 : 环绕通知 )
        """记录工具执行时间和结果"""

        def wrapper(query):  # query代表工具的输入参数
            """记录工具执行时间和结果"""
            logger.info(f"[{node_name}] 工具开始执行")
            start = time.time()

            result = func(query)  # 执行节点函数

            elapsed = time.time() - start
            logger.info(f"[{node_name}] 工具完成, 耗时 {elapsed:.2f}s")
            # 记录结果摘要
            if isinstance(result, dict):  # 说明当前执行的节点的返回值是一个字典
                for key, value in result.items():
                    if isinstance(value, str):
                        logger.info(f"[{node_name}] 输出 {key}: {len(value)} 字符")
                    elif isinstance(value, list):
                        logger.info(f"[{node_name}] 输出 {key}: {len(value)} 项")
            else:  # 这个节点是路由节点, 返回值是一个字符串
                logger.info(f"[{node_name}] 工具输出: {result}")
            return result

        return wrapper

    return decorator
