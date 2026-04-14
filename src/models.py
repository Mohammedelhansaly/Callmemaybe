from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, RootModel, model_validator

SupportedType = Literal["string", "number", "boolean"]


class FunctionReturn(BaseModel):
    type: SupportedType


class FunctionParameter(BaseModel):
    type: SupportedType


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, FunctionParameter]
    returns: FunctionReturn

    @model_validator(mode="after")
    def validate_parameters(self) -> "FunctionDefinition":
        for param_name in self.parameters:
            if not param_name.strip():
                raise ValueError("parameter name must not be empty")
        return self


class PromptItem(BaseModel):
    prompt: str = Field(..., min_length=1)


class FunctionCallResult(BaseModel):
    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parameters: Dict[str, Any]

    @classmethod
    def from_definition(
        cls,
        prompt: str,
        function_def: FunctionDefinition,
        parameters: Dict[str, Any],
    ) -> "FunctionCallResult":
        cls.validate_parameters_against_definition(function_def, parameters)
        return cls(
            prompt=prompt,
            name=function_def.name,
            parameters=parameters,
        )

    @staticmethod
    def validate_parameters_against_definition(
        function_def: FunctionDefinition,
        parameters: Dict[str, Any],
    ) -> None:
        expected_params = function_def.parameters

        if set(parameters.keys()) != set(expected_params.keys()):
            raise ValueError(
                "Provided parameters do not exactly match function definition."
            )

        for param_name, param_def in expected_params.items():
            value = parameters[param_name]
            param_type = param_def.type

            if param_type == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{param_name} must be string")

            elif param_type == "number":
                if (
                    not isinstance(value,
                                   (float, int)) or isinstance(value, bool)
                ):
                    raise ValueError(f"{param_name} must be number")

            elif param_type == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(f"{param_name} must be boolean")

            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")


class FunctionDefinitionFile(RootModel[List[FunctionDefinition]]):
    pass


class PromptItemsFile(RootModel[List[PromptItem]]):
    pass


def validate_function_definitions(data: Any) -> List[FunctionDefinition]:
    validated = FunctionDefinitionFile.model_validate(data)
    return validated.root


def validate_prompt_items(data: Any) -> List[PromptItem]:
    validated = PromptItemsFile.model_validate(data)
    return validated.root
