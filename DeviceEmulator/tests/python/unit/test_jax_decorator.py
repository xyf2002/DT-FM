import jax
import jaxlib
import jaxlib.xla_extension

pyobj_keywords = [
    "__class__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
    "Array",
]


def jax_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        for arg in args:
            print(type(arg))
        print("jax_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


# for all functions in jax, add decorator
functions = [f for f in dir(jax) if callable(getattr(jax, f))]
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(jax, f)
    setattr(jax, f, jax_decorator(func, f))

# for all functions in jax.numpy, add decorator
functions = [f for f in dir(jax.numpy) if callable(getattr(jax.numpy, f))]
print(functions)
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(jax.numpy, f)
    setattr(jax.numpy, f, jax_decorator(func, f))

# for all functions in jax.Array, add decorator
functions = [
    f
    for f in dir(jaxlib.xla_extension.ArrayImpl)
    if callable(getattr(jaxlib.xla_extension.ArrayImpl, f))
]
print(functions)
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(jaxlib.xla_extension.ArrayImpl, f)
    setattr(jaxlib.xla_extension.ArrayImpl, f, jax_decorator(func, f))


a = jax.numpy.ones((1, 3))
b = jax.numpy.ones((1, 3))

print(type(a), dir(a))
print(isinstance(a, jax.Array))

print(a * b)
print(a**2)
