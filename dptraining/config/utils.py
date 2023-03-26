from typing import Type


def get_allowed_names(enum_class: Type):
    return [member.name for member in enum_class]


def get_allowed_values(enum_class: Type):
    return [member.value for member in enum_class]
