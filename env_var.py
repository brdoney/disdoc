import os


def load_env(
    env_var: str, default: str | None = None, choices: list[str] | None = None
) -> str:
    if default is not None:
        res = os.getenv(env_var, default)
        return res

    val = os.getenv(env_var)

    if val is None:
        raise ValueError(
            f"Environment variable {env_var} must be defined in .env or otherwise."
        )
    if choices is not None and val not in choices:
        raise ValueError(
            f"Value {val} for {env_var} not valid. Valid choices are: {choices}"
        )
    return val
