"""Generate the API reference pages for the documentation.

Reads the numpydoc docstrings in ``src/cesm_hawc`` by parsing the source
(nothing is imported, so the ``[sim]`` dependencies are not needed) and
writes one Markdown page per module to ``docs/reference/api/``. Run it
before building the book::

    python docs/generate_api.py
    cd docs && jupyter book build --html

The output directory is regenerated from scratch on every run and is not
committed.

Each object links to its source on GitHub. In GitHub Actions the links point
at the commit being built (``GITHUB_SHA``); elsewhere they point at ``main``.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "cesm_hawc"
SRC = REPO / "src" / PACKAGE
OUT = REPO / "docs" / "reference" / "api"

# Section headers rendered as lists of "name : type / description" entries.
ENTRY_SECTIONS = {"Parameters", "Other Parameters", "Returns", "Yields", "Raises",
                  "Warns", "Attributes"}
SECTION_ORDER = ["Parameters", "Other Parameters", "Returns", "Yields", "Raises",
                 "Warns", "Attributes", "See Also", "Notes", "References", "Examples"]
MAX_SIGNATURE_WIDTH = 80

# Source links: the repository and commit being built in GitHub Actions,
# otherwise the main branch of the project repository.
if os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_REPOSITORY"):
    GITHUB_URL = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
else:
    GITHUB_URL = "https://github.com/torimcd/cesm-hawc"
GIT_REF = os.environ.get("GITHUB_SHA", "main")


def source_url(path: str, lines: tuple[int, int] | None = None) -> str:
    """GitHub URL for a repository file, optionally highlighting lines."""
    url = f"{GITHUB_URL}/blob/{GIT_REF}/{path}"
    if lines:
        url += f"#L{lines[0]}-L{lines[1]}"
    return url


# ---------------------------------------------------------------------------
# Collecting objects from the source
# ---------------------------------------------------------------------------

@dataclass
class Obj:
    """A documented function, method or class."""
    qualname: str              # e.g. "WACCMAtmosphere.get_column_profiles"
    module: str                # e.g. "cesm_hawc.waccm"
    kind: str                  # "function", "method", "property", "class"
    signature: str
    doc: str
    source: str = ""           # repository-relative path of the file
    lines: tuple[int, int] = (0, 0)
    methods: list["Obj"] = field(default_factory=list)

    @property
    def fqname(self) -> str:
        return f"{self.module}.{self.qualname}"


def label_for(fqname: str) -> str:
    """Cross-reference label for a fully qualified name."""
    return "api-" + fqname.lower().replace(".", "-")


def _format_args(args: ast.arguments, drop_first: bool) -> list[str]:
    """Render function arguments as ``name: annotation = default`` strings."""
    def one(a: ast.arg, default: ast.expr | None, prefix: str = "") -> str:
        text = prefix + a.arg
        if a.annotation is not None:
            text += f": {ast.unparse(a.annotation)}"
        if default is not None:
            text += (" = " if a.annotation is not None else "=") + ast.unparse(default)
        return text

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = [one(a, d) for a, d in zip(positional, defaults)]
    if args.posonlyargs:
        parts.insert(len(args.posonlyargs), "/")
    if drop_first and parts:
        parts.pop(0)
    if args.vararg:
        parts.append(one(args.vararg, None, "*"))
    elif args.kwonlyargs:
        parts.append("*")
    parts += [one(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        parts.append(one(args.kwarg, None, "**"))
    return parts


def _signature(keyword: str, name: str, parts: list[str], returns: str | None) -> str:
    """One-line signature, or one parameter per line if too long."""
    suffix = f" -> {returns}" if returns else ""
    flat = f"{keyword} {name}({', '.join(parts)}){suffix}"
    if len(flat) <= MAX_SIGNATURE_WIDTH or not parts:
        return flat
    body = "".join(f"    {p},\n" for p in parts)
    return f"{keyword} {name}(\n{body}){suffix}"


def _function(node: ast.FunctionDef, module: str, owner: str | None,
              source: str) -> Obj | None:
    if node.name.startswith("_"):
        return None
    decorators = {ast.unparse(d) for d in node.decorator_list}
    is_property = "property" in decorators
    is_method = owner is not None and "staticmethod" not in decorators
    parts = _format_args(node.args, drop_first=is_method)
    returns = ast.unparse(node.returns) if node.returns is not None else None
    qualname = f"{owner}.{node.name}" if owner else node.name
    if is_property:
        sig = f"property {node.name}" + (f" -> {returns}" if returns else "")
        kind = "property"
    else:
        keyword = "classmethod" if "classmethod" in decorators else "def"
        sig = _signature(keyword, node.name, parts, returns)
        kind = "method" if owner else "function"
    return Obj(qualname, module, kind, sig, ast.get_docstring(node) or "",
               source, (node.lineno, node.end_lineno))


def _class(node: ast.ClassDef, module: str, source: str) -> Obj | None:
    if node.name.startswith("_"):
        return None
    is_dataclass = any("dataclass" in ast.unparse(d) for d in node.decorator_list)
    init = next((n for n in node.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is not None:
        parts = _format_args(init.args, drop_first=True)
    elif is_dataclass:
        parts = []
        for n in node.body:
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                text = f"{n.target.id}: {ast.unparse(n.annotation)}"
                if n.value is not None:
                    text += f" = {ast.unparse(n.value)}"
                parts.append(text)
    else:
        parts = []
    bases = [ast.unparse(b) for b in node.bases]
    sig = _signature("class", node.name, parts, None) if (parts or init) else (
        f"class {node.name}" + (f"({', '.join(bases)})" if bases else ""))
    obj = Obj(node.name, module, "class", sig, ast.get_docstring(node) or "",
              source, (node.lineno, node.end_lineno))
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            method = _function(n, module, owner=node.name, source=source)
            if method is not None:
                obj.methods.append(method)
    return obj


def collect(path: Path) -> tuple[str, str, list[Obj]]:
    """Return ``(module name, module docstring, public objects)`` for a file."""
    module = PACKAGE if path.name == "__init__.py" else f"{PACKAGE}.{path.stem}"
    tree = ast.parse(path.read_text())
    source = path.relative_to(REPO).as_posix()
    objs: list[Obj] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            obj = _function(node, module, owner=None, source=source)
        elif isinstance(node, ast.ClassDef):
            obj = _class(node, module, source)
        else:
            obj = None
        if obj is not None:
            objs.append(obj)
    return module, ast.get_docstring(tree) or "", objs


# ---------------------------------------------------------------------------
# Converting numpydoc / reST to MyST Markdown
# ---------------------------------------------------------------------------

_ROLE_RE = re.compile(r":(?:py:)?(func|meth|class|mod|data|attr|obj|exc):`(~?)([^`]+)`")
_LITERAL_RE = re.compile(r"``(.+?)``")


class Linker:
    """Resolves reST roles such as :func:`name` to links between API pages."""

    def __init__(self, targets: set[str]):
        self.targets = targets

    def resolve(self, name: str, module: str, owner: str | None) -> str | None:
        candidates = [name]
        if owner:
            candidates.append(f"{module}.{owner}.{name}")
        candidates.append(f"{module}.{name}")
        candidates.append(f"{PACKAGE}.{name}")
        found = next((c for c in candidates if c in self.targets), None)
        if found is None:
            # e.g. a name re-exported from the package's __init__
            matches = [t for t in self.targets if t.endswith("." + name)]
            if len(matches) == 1:
                found = matches[0]
        return found

    def inline(self, text: str, module: str, owner: str | None) -> str:
        def role(m: re.Match) -> str:
            kind, tilde, name = m.groups()
            target = self.resolve(name, module, owner)
            shown = name.split(".")[-1] if tilde else name
            if kind in ("func", "meth"):
                shown += "()"
            if target is None:
                return f"`{shown}`"
            return f"[`{shown}`](#{label_for(target)})"

        text = _ROLE_RE.sub(role, text)
        return _LITERAL_RE.sub(lambda m: f"`{m.group(1)}`", text)


def split_sections(doc: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split a numpydoc docstring into its free text and named sections."""
    lines = inspect.cleandoc(doc).splitlines()
    intro: list[str] = []
    sections: dict[str, list[str]] = {}
    current = intro
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (line.strip() and nxt.strip() and set(nxt.strip()) == {"-"}
                and len(nxt.strip()) >= len(line.strip()) - 1 and not line.startswith(" ")):
            current = sections.setdefault(line.strip(), [])
            i += 2
            continue
        current.append(line)
        i += 1
    return intro, sections


def literal_blocks(lines: list[str]) -> list[str]:
    """Turn reST ``::`` literal blocks into fenced code blocks."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            stripped = line.rstrip()[:-2].rstrip()
            if stripped:
                out.append(stripped + ":")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out += ["", "```", *inspect.cleandoc("\n".join(block)).splitlines(), "```", ""]
            continue
        out.append(line)
        i += 1
    return out


def render_text(lines: list[str], linker: Linker, module: str, owner: str | None) -> str:
    """Render free text: literal blocks fenced, inline markup converted."""
    out = []
    in_fence = False
    for line in literal_blocks(lines):
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(linker.inline(line, module, owner))
    return "\n".join(out).strip()


def render_entries(name: str, lines: list[str], linker: Linker, module: str,
                   owner: str | None) -> str:
    """Render a Parameters-style section as a bulleted list."""
    entries: list[tuple[str, list[str]]] = []
    for line in lines:
        if line and not line.startswith((" ", "\t")):
            entries.append((line.strip(), []))
        elif entries:
            entries[-1][1].append(line)
    items = []
    for header, body in entries:
        if " : " in header:
            names, typ = header.split(" : ", 1)
        elif name in ("Parameters", "Other Parameters", "Attributes"):
            names, typ = header, ""
        else:
            names, typ = "", header
        if names:
            head = ", ".join(f"**`{n.strip()}`**" for n in names.split(","))
            if typ:
                head += f" ({linker.inline(typ, module, owner)})"
        else:
            head = f"*{linker.inline(typ, module, owner)}*" if name not in ("Raises", "Warns") \
                else f"`{typ}`"
        desc = render_text(inspect.cleandoc("\n".join(body)).splitlines(), linker, module, owner)
        if desc:
            desc_lines = desc.splitlines()
            first = desc_lines[0]
            rest = "".join(f"\n  {d}" if d.strip() else "\n" for d in desc_lines[1:])
            items.append(f"- {head} — {first}{rest}")
        else:
            items.append(f"- {head}")
    return "\n".join(items)


def render_examples(lines: list[str], linker: Linker, module: str, owner: str | None) -> str:
    """Render an Examples section, fencing doctest lines as Python."""
    out: list[str] = []
    in_code = False
    for line in lines:
        is_code = line.startswith((">>>", "...")) or (in_code and line.strip())
        if is_code and not in_code:
            out += ["", "```python"]
            in_code = True
        elif not is_code and in_code:
            out += ["```", ""]
            in_code = False
        out.append(line if in_code else linker.inline(line, module, owner))
    if in_code:
        out.append("```")
    return "\n".join(out).strip()


def render_doc(doc: str, linker: Linker, module: str, owner: str | None,
               heading: str = "####") -> str:
    intro, sections = split_sections(doc)
    parts = [render_text(intro, linker, module, owner)]
    names = [s for s in SECTION_ORDER if s in sections] + \
            [s for s in sections if s not in SECTION_ORDER]
    for name in names:
        lines = sections[name]
        if name in ENTRY_SECTIONS:
            body = render_entries(name, lines, linker, module, owner)
        elif name == "Examples":
            body = render_examples(lines, linker, module, owner)
        else:
            body = render_text(lines, linker, module, owner)
        parts.append(f"{heading} {name}\n\n{body}")
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Writing pages
# ---------------------------------------------------------------------------

def render_object(obj: Obj, linker: Linker, level: int) -> str:
    owner = obj.qualname.split(".")[0] if obj.kind in ("method", "property", "class") else None
    head = "#" * level
    parts = [
        f"({label_for(obj.fqname)})=",
        f"{head} `{obj.qualname}`",
        "",
        "```python",
        obj.signature,
        "```",
        "",
        f"[source]({source_url(obj.source, obj.lines)})",
        "",
        render_doc(obj.doc, linker, obj.module, owner, heading="#" * (level + 1)) if obj.doc
        else "*Not documented.*",
    ]
    for method in obj.methods:
        parts += ["", render_object(method, linker, level + 1)]
    return "\n".join(parts)


def summary_line(doc: str) -> str:
    return inspect.cleandoc(doc).splitlines()[0] if doc.strip() else ""


def main() -> None:
    modules = []
    for path in sorted(SRC.glob("*.py")):
        modules.append(collect(path))

    targets: set[str] = set()
    for module, _, objs in modules:
        targets.add(module)
        for obj in objs:
            targets.add(obj.fqname)
            targets.update(m.fqname for m in obj.methods)
    linker = Linker(targets)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    header = ("<!-- Generated by docs/generate_api.py from the docstrings in "
              "src/cesm_hawc. Do not edit. -->\n\n")
    rows = []
    package_doc = ""
    for module, doc, objs in modules:
        if module == PACKAGE:
            package_doc = doc
            continue
        page = OUT / f"{module}.md"
        body = [
            header + f"({label_for(module)})=",
            f"# `{module}`",
            "",
            f"[source]({source_url(f'src/{PACKAGE}/{module.rsplit(chr(46), 1)[1]}.py')})",
            "",
            render_doc(doc, linker, module, None, heading="##"),
        ]
        for obj in objs:
            body += ["", render_object(obj, linker, level=2)]
        page.write_text("\n".join(body).rstrip() + "\n")
        rows.append(f"| [`{module}`](#{label_for(module)}) | "
                    f"{linker.inline(summary_line(doc), module, None)} |")

    index = [
        header + f"({label_for(PACKAGE)})=",
        "# Python API",
        "",
        render_doc(package_doc, linker, PACKAGE, None, heading="##"),
        "",
        "| Module | Description |",
        "|--------|-------------|",
        *rows,
    ]
    (OUT / "index.md").write_text("\n".join(index) + "\n")
    print(f"Wrote {len(rows) + 1} pages to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
