import pytest

import json
import nbformat
import os
import re
from nbconvert.preprocessors import ExecutePreprocessor

notebook_timeout_seconds = 3600
notebook_dir = "./finn_examples/notebooks/"


# Needed for iterating through each mnist and cifar10 model
# within their respective notebooks, formatted as:
# models = [
#   (model_name, expected_accuracy),
#   ...
# ]
mnist_models = [
    ("tfc_w1a1_mnist", 92.96),
    ("tfc_w1a2_mnist", 94.74),
    ("tfc_w2a2_mnist", 96.6),
]

cifar10_models = [
    ("cnv_w1a1_cifar10", 84.19),
    ("cnv_w1a2_cifar10", 87.76),
    ("cnv_w2a2_cifar10", 88.63),
]

# List of all notebooks to be tested, formatted as:
# all_notebooks = [
#       (
#       notebook_name,
#       model_name,
#       expected_returned_label,
#       expected_accuracy,
#       )
#       ...
#   ]

all_notebooks = [
    (
        # model name and accuracy will be assigned using mnist_models during test
        notebook_dir + "/0_mnist_with_fc_networks.ipynb",
        " ",
        "Returned class is 7",
        " ",
    ),
    (
        # model name and accuracy will be assigned using cifar10_models during test
        notebook_dir + "/1_cifar10_with_cnv_networks.ipynb",
        " ",
        "Returned class is 3",
        " ",
    ),
    (
        notebook_dir + "/2_imagenet_with_cnns.ipynb",
        "mobilenetv1_w4a4_imagenet",
        " ",
        70.406,
    ),
    (
        notebook_dir + "/3_binarycop_mask_detection.ipynb",
        "bincop_cnv",
        " ",
        " ",
    ),
    (
        notebook_dir + "/4_keyword_spotting.ipynb",
        "kws_mlp",
        "The audio file was classified as: yes",
        88.7646,
    ),
    (
        notebook_dir + "/5_radioml_with_cnns.ipynb",
        "vgg_w4a4_radioml",
        "Top-1 class predicted by the accelerator: 16QAM",
        87.886,
    ),
    (
        notebook_dir + "/6_cybersecurity_with_mlp.ipynb",
        "mlp_w2a2_unsw_nb15",
        "Returned label is: 0 (normal data)",
        91.90,
    ),
    (
        notebook_dir + "/7_traffic_sign_recognition_gtsrb.ipynb",
        "cnv_w1a1_gtsrb",
        "Accelerator result is:\nProhibited for vehicles with a "
        "permitted gross weight over 3.5t including their trailers, "
        "and for tractors except passenger cars and buses",
        94.9485,
    ),
]

# List of notebooks for each platform
pynq_notebooks = [
    # 0_mnist_with_fc_networks.ipynb
    pytest.param(
        all_notebooks[0][0],
        all_notebooks[0][1],
        all_notebooks[0][2],
        all_notebooks[0][3],
        marks=pytest.mark.pynq_notebooks,
    ),
    # 1_cifar10_with_cnv_networks.ipynb
    pytest.param(
        all_notebooks[1][0],
        all_notebooks[1][1],
        all_notebooks[1][2],
        all_notebooks[1][3],
        marks=pytest.mark.pynq_notebooks,
    ),
    # 3_binarycop_mask_detection.ipynb
    pytest.param(
        all_notebooks[3][0],
        all_notebooks[3][1],
        all_notebooks[3][2],
        all_notebooks[3][3],
        marks=[pytest.mark.pynq_notebooks, pytest.mark.xfail],
    ),
    # 4_keyword_spotting.ipynb
    pytest.param(
        all_notebooks[4][0],
        all_notebooks[4][1],
        all_notebooks[4][2],
        all_notebooks[4][3],
        marks=pytest.mark.pynq_notebooks,
    ),
    # 6_cybersecurity_with_mlp.ipynb
    pytest.param(
        all_notebooks[6][0],
        all_notebooks[6][1],
        all_notebooks[6][2],
        all_notebooks[6][3],
        marks=pytest.mark.pynq_notebooks,
    ),
    # 7_traffic_sign_recognition_gtsrb.ipynb
    pytest.param(
        all_notebooks[7][0],
        all_notebooks[7][1],
        all_notebooks[7][2],
        all_notebooks[7][3],
        marks=pytest.mark.pynq_notebooks,
    ),
]

zcu_notebooks = [
    # 0_mnist_with_fc_networks.ipynb
    pytest.param(
        all_notebooks[0][0],
        all_notebooks[0][1],
        all_notebooks[0][2],
        all_notebooks[0][3],
        marks=pytest.mark.zcu_notebooks,
    ),
    # 1_cifar10_with_cnv_networks.ipynb
    pytest.param(
        all_notebooks[1][0],
        all_notebooks[1][1],
        all_notebooks[1][2],
        all_notebooks[1][3],
        marks=pytest.mark.zcu_notebooks,
    ),
    # 5_radioml_with_cnns.ipynb
    pytest.param(
        all_notebooks[5][0],
        all_notebooks[5][1],
        all_notebooks[5][2],
        all_notebooks[5][3],
        marks=pytest.mark.zcu_notebooks,
    ),
    # 6_cybersecurity_with_mlp.ipynb
    pytest.param(
        all_notebooks[6][0],
        all_notebooks[6][1],
        all_notebooks[6][2],
        all_notebooks[6][3],
        marks=pytest.mark.zcu_notebooks,
    ),
]

ultra96_notebooks = [
    # 0_mnist_with_fc_networks.ipynb
    pytest.param(
        all_notebooks[0][0],
        all_notebooks[0][1],
        all_notebooks[0][2],
        all_notebooks[0][3],
        marks=pytest.mark.ultra96_notebooks,
    ),
    # 1_cifar10_with_cnv_networks.ipynb
    pytest.param(
        all_notebooks[1][0],
        all_notebooks[1][1],
        all_notebooks[1][2],
        all_notebooks[1][3],
        marks=pytest.mark.ultra96_notebooks,
    ),
    # 6_cybersecurity_with_mlp.ipynb
    pytest.param(
        all_notebooks[6][0],
        all_notebooks[6][1],
        all_notebooks[6][2],
        all_notebooks[6][3],
        marks=pytest.mark.ultra96_notebooks,
    ),
]

alveo_notebooks = [
    # 0_mnist_with_fc_networks.ipynb
    pytest.param(
        all_notebooks[0][0],
        all_notebooks[0][1],
        all_notebooks[0][2],
        all_notebooks[0][3],
        marks=pytest.mark.alveo_notebooks,
    ),
    # 1_cifar10_with_cnv_networks.ipynb
    pytest.param(
        all_notebooks[1][0],
        all_notebooks[1][1],
        all_notebooks[1][2],
        all_notebooks[1][3],
        marks=pytest.mark.alveo_notebooks,
    ),
    # 2_imagenet_with_cnns.ipynb
    pytest.param(
        all_notebooks[2][0],
        all_notebooks[2][1],
        all_notebooks[2][2],
        all_notebooks[2][3],
        marks=[pytest.mark.alveo_notebooks, pytest.mark.xfail],
    ),
]


def get_notebook_exec_result(notebook, model_name, exp_label, exp_acc):
    # Read and execute the notebook
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=notebook_timeout_seconds, kernel_name="python3")
    ep.preprocess(nb)

    # Read in the executed notebook as a json
    exec_notebook = notebook.replace(".ipynb", "_exec.ipynb")
    with open(exec_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    with open(exec_notebook) as f:
        test_json_exec = json.load(f)

    # For checking if the correct class was predicted
    res = False

    # Get outputs of notebook json
    for cell in test_json_exec["cells"]:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "text" in output:
                    out_text = "".join(output["text"])
                    if exp_label in out_text:
                        # Expected class was predicted
                        res = True
                    if "accuracy" in out_text.casefold():
                        # Parse the accuracy value and check if it is as expected
                        nb_acc = float(re.findall("\\d+\\.\\d+", out_text)[-1])
                        assert nb_acc >= exp_acc, f"Accuracy test for {model_name} FAILED"

    assert res is True, f"Classification test for {model_name} FAILED"


@pytest.mark.parametrize(
    "notebook,model_name,exp_label,exp_acc",
    pynq_notebooks + zcu_notebooks + ultra96_notebooks + alveo_notebooks,
)
def test_notebook_exec(notebook, model_name, exp_label, exp_acc):
    if "mnist" in notebook:
        for mnist_model_name, mnist_exp_acc in mnist_models:
            os.system("sed -i '27s/.*/\"accel = models.%s()\"/' %s" % (mnist_model_name, notebook))
            get_notebook_exec_result(notebook, mnist_model_name, exp_label, mnist_exp_acc)
    elif "cifar10" in notebook:
        for cifar10_model_name, cifar10_exp_acc in cifar10_models:
            os.system(
                "sed -i '26s/.*/\"accel = models.%s()\"/' %s" % (cifar10_model_name, notebook)
            )
            get_notebook_exec_result(notebook, cifar10_model_name, exp_label, cifar10_exp_acc)
    else:
        get_notebook_exec_result(notebook, model_name, exp_label, exp_acc)

    os.system("rm -rf %s/*_exec*" % notebook_dir)
