# halfmesh Windows wheel

Built for CPython 3.10 on Windows x86-64 from commit `f218f7c1a41218cc71990bfec0b1557a2c1b60e8`. [Upstream source](https://github.com/cdcseacave/halfmesh/tree/f218f7c1a41218cc71990bfec0b1557a2c1b60e8)

The build uses Visual Studio Build Tools 2022, the `x64-windows-static-md` vcpkg triplet, and dependency baseline `0ac8df3b98e3afcd8bf075fa74a6bd2c32613345`. The halfmesh source is built from the pinned checkout.

Install with `uv pip install --python .venv/Scripts/python.exe --find-links third_party/wheels halfmesh==0.5.1` from the project root. The same wheel is referenced by requirements-lock.txt.

The halfmesh MIT license and bundled dependency licenses are in `licenses/`.

The wheel SHA-256 is `0e76727f9fa29a23e2d8f8bb964562670511de00bef346239c7dcb692a00890b`.
