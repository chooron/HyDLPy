from functools import wraps
from typing import TypeVar, Callable, Type
from typing_extensions import ParamSpec  # 需要 pip install typing_extensions（若 Python < 3.10）

P = ParamSpec('P')
T = TypeVar('T')

def static_identifier(id_str: str) -> Callable[[Callable[P, T]], staticmethod]:
    def decorator(func: Callable[P, T]) -> staticmethod:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            print(f"ID: {id_str}")
            return func(*args, **kwargs)
        return staticmethod(wrapper)
    return decorator

class MyClass:
    @static_identifier("math_op")
    def add(x: int, y: int) -> int:  # 无 self，IDE 不会报警告
        return x + y

# 测试
print(MyClass.add(2, 3))