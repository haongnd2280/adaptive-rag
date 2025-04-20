import time
from functools import wraps
from typing import Any, Callable, Literal, TypeAlias


Function: TypeAlias = Callable[..., Any]

def trace(
    component: Literal["edge", "node"],
    *,
    disable: bool = False,
) -> Function:
    
    def decorator(func: Function) -> Function:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not disable:
                print(f"@trace. {component}: {func.__name__}")

            result = func(*args, **kwargs)
            return result
        
        return wrapper
    return decorator


def timer(
    func: Callable[..., Any],
    *,
    disable: bool = False,
) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        if not disable:
            print(f"@timer. {func.__name__}. Runtime: {end - start:.2f}s.")

        return result

    return wrapper
