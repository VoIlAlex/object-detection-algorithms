import platform


def allow_MoyaPizama_constrains(cls):
    """
    Cleans None-valued method after
    MoyaPizama_specific_method decorator.

    This decorator allows class to 
    have method only available on 
    my PC. See  MoyaPizama_specific_method
    decorator docstring for more details.
    """
    if platform.node() == 'MoyaPizama':
        for attr in cls.__dict__:
            if attr.startswith('test_') and attr is None:
                delattr(cls, attr)
    return cls


def MoyaPizama_specific_method(func):
    """
    This decorator makes function 
    available only on my PC. This is
    useful for tests, because some tests
    consert data outside the project, 
    i.e. datasets. 
    """
    if platform.node() == 'MoyaPizama':
        return func
    else:
        return None
