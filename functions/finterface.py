"""
Interface definition
"""

from fdatastruct import fdatastruct
from fcontext import fcontext


def finterface1(d: fdatastruct, context: fcontext, **kwargs) -> fdatastruct:
    pass


def finterface2(d1: fdatastruct, d2: fdatastruct, context: fcontext, **kwargs) -> fdatastruct:
    pass


def finterface3(d1: fdatastruct, d2: fdatastruct, d3: fdatastruct, context: fcontext, **kwargs) -> fdatastruct:
    pass


def finterface4(d1: fdatastruct, d2: fdatastruct, d3: fdatastruct, d4: fdatastruct, context:
        fcontext, **kwargs) -> fdatastruct:
    pass

