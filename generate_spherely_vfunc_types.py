import itertools
import string
from pathlib import Path

STUB_FILE_PATH = Path(__file__).parent / "src" / "spherely.pyi"
BEGIN_MARKER = "# /// Begin types"
END_MARKER = "# /// End types"
LINE_PREFIX = "#     - "


def update_stub_file(path=STUB_FILE_PATH):
    stub_text = path.read_text(encoding="utf-8")
    try:
        start_idx = stub_text.index(BEGIN_MARKER)
        end_idx = stub_text.index(END_MARKER)
    except ValueError:
        raise SystemExit(
            f"Error: Markers '{BEGIN_MARKER}' and '{END_MARKER}' "
            f"were not found in stub file '{path}'"
        ) from None

    args_specs = [
        _parse_vfunctype_args(line.removeprefix(LINE_PREFIX))
        for line in stub_text[start_idx:end_idx].splitlines()
        if line.startswith(LINE_PREFIX)
    ]

    header = "\n".join(
        [BEGIN_MARKER, "#"]
        + [
            f"{LINE_PREFIX}{', '.join(f'{a}={t}' for a, t in args.items())}"
            for args in args_specs
        ]
        + ["#", ""]
    )
    code = "\n\n".join(_vfunctype_factory(**args) for args in args_specs)
    updated_stub_text = stub_text[:start_idx] + header + code + stub_text[end_idx:]
    path.write_text(updated_stub_text, encoding="utf-8")


def _parse_vfunctype_args(signature):
    types = {}
    for arg in signature.split(","):
        arg_name, _, arg_type = arg.strip().partition("=")
        types[arg_name.strip()] = arg_type.strip()

    # The `n_in` parameter isn't a type and should be interpreted as an int
    return types | {"n_in": int(types["n_in"])}


def _vfunctype_factory(n_in, **optargs):
    """Create new VFunc types.

    Based on the number of input arrays and optional arguments and their types."""
    names = ["geography"] if n_in == 1 else list(string.ascii_lowercase[:n_in])
    class_name = f"_VFunc_Nin{n_in}{''.join(optargs)}_Nout1"
    class_code = [
        f"class {class_name}(",
        "    Generic[_NameType, _ScalarReturnType, _ArrayReturnDType]",
        "):",
        "    @property",
        "    def __name__(self) -> _NameType: ...",
        "",
    ]
    optarg_str = ", ".join(
        f"{arg_name}: {arg_type} = ..." for arg_name, arg_type in optargs.items()
    )

    geog_types = ["Geography", "npt.ArrayLike"]
    for types in itertools.product(geog_types, repeat=n_in):
        arg_str = ", ".join(
            f"{arg_name}: {arg_type}" for arg_name, arg_type in zip(names, types)
        )
        return_type = (
            "_ScalarReturnType"
            if all(t == geog_types[0] for t in types)
            else "npt.NDArray[_ArrayReturnDType]"
        )
        class_code.extend(
            [
                "    @overload",
                "    def __call__(",
                (
                    f"        self, {arg_str}, {optarg_str}"
                    if optarg_str
                    else f"        self, {arg_str}"
                ),
                f"    ) -> {return_type}: ...",
                "",
            ]
        )
    return "\n".join(class_code)


if __name__ == "__main__":
    update_stub_file()
