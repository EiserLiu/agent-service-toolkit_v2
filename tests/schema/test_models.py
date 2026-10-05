"""模型目录必须满足的约束，get_model 的分派逻辑依赖这些约束。"""

import typing
from collections import defaultdict

import pytest

from core.llm import _MODEL_TABLE
from schema.models import AllModelEnum, GoogleModelName, VertexAIModelName

MODEL_ENUMS = typing.get_args(AllModelEnum.__value__)


def test_model_enums_are_all_discovered():
    # 空元组会使下方测试在没有实际检查任何内容时直接通过。
    assert len(MODEL_ENUMS) >= 12


def test_model_values_are_unique_across_enums():
    # StrEnum 按值计算哈希，因此同值成员在 _MODEL_TABLE 和
    # AVAILABLE_MODELS 中会合并为一项，get_model 也会将两者都路由到第一个分支。
    owners = defaultdict(list)
    for enum in MODEL_ENUMS:
        for member in enum:
            owners[member.value].append(f"{enum.__name__}.{member.name}")

    duplicates = {value: names for value, names in owners.items() if len(names) > 1}
    assert not duplicates, f"model values must be unique across enums: {duplicates}"


@pytest.mark.parametrize("model", list(VertexAIModelName), ids=lambda m: m.value)
def test_vertexai_models_do_not_collide_with_google(model):
    # 每个 Vertex 模型都有对应的 Google 模型，此处值冲突会创建错误的客户端。
    assert model not in GoogleModelName


def test_model_table_covers_every_model():
    # _MODEL_TABLE 与 AllModelEnum 联合类型分别维护。
    assert set(_MODEL_TABLE) == {member for enum in MODEL_ENUMS for member in enum}
