import pytest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_timeout_seconds = 3600
notebook_dir = "./finn_examples/notebooks/"

pynq_notebooks = [
    pytest.param(
        notebook_dir + "0_mnist_with_fc_networks.ipynb",
        marks=pytest.mark.pynq_notebooks,
    ),
    pytest.param(
        notebook_dir + "1_cifar10_with_cnv_networks.ipynb",
        marks=pytest.mark.pynq_notebooks,
    ),
    pytest.param(
        notebook_dir + "3_binarycop_mask_detection.ipynb",
        marks=[pytest.mark.pynq_notebooks, pytest.mark.xfail],
    ),
    pytest.param(
        notebook_dir + "4_keyword_spotting.ipynb",
        marks=pytest.mark.pynq_notebooks,
    ),
    pytest.param(
        notebook_dir + "6_cybersecurity_with_mlp.ipynb",
        marks=pytest.mark.pynq_notebooks,
    ),
]

zcu_notebooks = [
    pytest.param(
        notebook_dir + "0_mnist_with_fc_networks.ipynb",
        marks=pytest.mark.zcu_notebooks,
    ),
    pytest.param(
        notebook_dir + "1_cifar10_with_cnv_networks.ipynb",
        marks=pytest.mark.zcu_notebooks,
    ),
    pytest.param(
        notebook_dir + "5_radioml_with_cnns.ipynb",
        marks=pytest.mark.zcu_notebooks,
    ),
    pytest.param(
        notebook_dir + "6_cybersecurity_with_mlp.ipynb",
        marks=pytest.mark.zcu_notebooks,
    ),
]

ultra96_notebooks = [
    pytest.param(
        notebook_dir + "0_mnist_with_fc_networks.ipynb",
        marks=pytest.mark.ultra96_notebooks,
    ),
    pytest.param(
        notebook_dir + "1_cifar10_with_cnv_networks.ipynb",
        marks=pytest.mark.ultra96_notebooks,
    ),
    pytest.param(
        notebook_dir + "6_cybersecurity_with_mlp.ipynb",
        marks=pytest.mark.ultra96_notebooks,
    ),
]

alveo_notebooks = [
    pytest.param(
        notebook_dir + "0_mnist_with_fc_networks.ipynb",
        marks=pytest.mark.alveo_notebooks,
    ),
    pytest.param(
        notebook_dir + "1_cifar10_with_cnv_networks.ipynb",
        marks=pytest.mark.alveo_notebooks,
    ),
    pytest.param(
        notebook_dir + "2_imagenet_with_cnns.ipynb",
        marks=pytest.mark.alveo_notebooks,
    ),
]


@pytest.mark.parametrize(
    "notebook", pynq_notebooks + zcu_notebooks + ultra96_notebooks + alveo_notebooks
)
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=notebook_timeout_seconds, kernel_name="python3")

        # debug only for now...
        notebook_dump = notebook.replace(".ipynb", ".dump")
        with open(notebook_dump, "w") as f:
            f.write(str(ep.preprocess(nb)))

        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"
