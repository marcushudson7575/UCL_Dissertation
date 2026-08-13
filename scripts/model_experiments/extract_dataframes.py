"""Execute a notebook headlessly (in its own dir) with an appended cell that
pickles the modelling dataframes to the scratchpad. Usage:
    python extract_dataframes.py <notebook_path> <pickle_out> <var1> <var2> ...
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path = Path(sys.argv[1])
pickle_out = sys.argv[2]
varnames = sys.argv[3:]

nb = nbformat.read(nb_path, as_version=4)

# Neutralise plotly browser rendering and force matplotlib Agg
prelude = nbformat.v4.new_code_cell(
    "import matplotlib\nmatplotlib.use('Agg')\n"
    "try:\n    import plotly.io as _pio\n    _pio.renderers.default = 'json'\nexcept Exception:\n    pass\n"
)
dump = nbformat.v4.new_code_cell(
    "import pickle\n"
    f"_targets = {varnames!r}\n"
    "_found = {k: v for k, v in list(globals().items()) if k in _targets}\n"
    f"with open({pickle_out!r}, 'wb') as _f:\n"
    "    pickle.dump(_found, _f)\n"
    "print('PICKLED:', sorted(_found), 'MISSING:', sorted(set(_targets) - set(_found)))\n"
)
nb.cells = [prelude] + nb.cells + [dump]

client = NotebookClient(
    nb,
    timeout=1800,
    kernel_name="python3",
    resources={"metadata": {"path": str(nb_path.parent)}},
)
client.execute()

# print output of the last cell so the caller sees PICKLED/MISSING
for out in nb.cells[-1].get("outputs", []):
    if out.get("output_type") == "stream":
        print(out["text"])
print("EXECUTION COMPLETE")
