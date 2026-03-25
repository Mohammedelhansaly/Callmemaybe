from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Literal, Dict, Any

supportedType = Literal["string", "number", "boolean"]


class FunctionReturn(BaseModel):
    type: supportedType


class FunctionParameter(BaseModel):
    type: supportedType


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parametrs: Dict[str, FunctionParameter]
    returns: FunctionReturn

    @model_validator(mode='after')
    def validate_parameters(self):
        for param_name in self.parametrs:
            if not param_name.strip():
                raise ValueError("parameter must not be empty")


class PromptItem(BaseModel):
    prompt: str = Field(..., min_length=1)


class FunctionCallResult(BaseModel):
    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parametrs: Dict[str, Any]

    @classmethod
    def from_definition(cls, prompt,
                        function_def, parameters) -> "FunctionCallResult":
        cls.validate_parametrs_against_definition(function_def, parameters)
        return cls(
            prompt=prompt,
            name=function_def,
            parameters=parameters
        )

    @classmethod
    def validate_parametrs_against_definition(function_def, paramerters):
        excepted_params = function_def.parametrs
        if set(paramerters.keys()) != set(excepted_params.keys()):
            raise ValueError("Provided parameters do not exactly "
                             "match function definition.")
        for param_name, param_def in excepted_params.items()
            value = paramerters[param_name]
            param_type = param_def.type
            if param_type == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{param_name} must be string")
