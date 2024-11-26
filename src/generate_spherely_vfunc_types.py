import itertools
import string
from pathlib import Path

VFUNC_TYPE_SPECS = {
    "_VFunc_Nin1_Nout1": {"n_in": 1},
    "_VFunc_Nin2_Nout1": {"n_in": 2},
    "_VFunc_Nin2optradius_Nout1": {"n_in": 2, "radius": "float"},
    "_VFunc_Nin1optradius_Nout1": {"n_in": 1, "radius": "float"},
}

STUB_FILE_PATH = Path(__file__).parent / "spherely.pyi"
BEGIN_MARKER = "# /// Begin types"
END_MARKER = "# /// End types"


def update_stub_file(path, **type_specs):
    stub_text = path.read_text(encoding="utf-8")
    try:
        start_idx = stub_text.index(BEGIN_MARKER)
        end_idx = stub_text.index(END_MARKER)
    except ValueError:
        raise SystemExit(
            f"Error: Markers '{BEGIN_MARKER}' and '{END_MARKER}' "
            f"were not found in stub file '{path}'"
        ) from None

    header = f"{BEGIN_MARKER}\n"
    code = "\n\n".join(
        _vfunctype_factory(name, **args) for name, args in type_specs.items()
    )
    updated_stub_text = stub_text[:start_idx] + header + code + stub_text[end_idx:]
    path.write_text(updated_stub_text, encoding="utf-8")


def _vfunctype_factory(class_name, n_in, **optargs):
    """Create new VFunc types.

    Based on the number of input arrays and optional arguments and their types."""
    arg_names = (
        ["geography"]
        if n_in == 1 and not optargs
        else list(string.ascii_lowercase[:n_in])
    )
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
    for arg_types in itertools.product(geog_types, repeat=n_in):
        arg_str = ", ".join(
            f"{arg_name}: {arg_type}"
            for arg_name, arg_type in zip(arg_names, arg_types)
        )
        return_type = (
            "_ScalarReturnType"
            if all(t == geog_types[0] for t in arg_types)
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
    update_stub_file(path=STUB_FILE_PATH, **VFUNC_TYPE_SPECS)
