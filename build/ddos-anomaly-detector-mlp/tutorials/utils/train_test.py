#!/usr/bin/env python3
"""
Common utility functions related to training and testing models.
"""

import torch
from typing import List
from sklearn.metrics import accuracy_score
from tqdm import trange
import finn.core.onnx_exec as oxe


def train(model, train_loader, optimizer, criterion, device: str = "cpu") -> List[float]:
    """
    Runs one epoch of training the model.

    Args:

        model: Input model to train.

        train_loader: PyTorch DataLoader wrapper that provides batches of input
        data to train the model with.

        optimizer: Optimizer to use during each training iteration.

        criterion: Loss function at the output

        device (str): Device name on which to train the model. Default = "cpu".

    Returns:

        List[float]: List of losses after each iteration in the epoch.
    """
    # collect the losses after each iteration
    losses = []

    # ensure model is in training mode
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(inputs.float())
        loss = criterion(output, target.unsqueeze(1))

        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.data.cpu().numpy())

    return losses


def test(model, test_loader, device: str = "cpu", bipolar: bool = False) -> float:
    """
    Runs inference on model with test set and reports the accuracy achieved.

    Args:

        model: Input model to run inference on.

        test_loader: PyTorch DataLoader wrapper that provides batches of input
        data to run inference on.

        device (str): Device name on which to run the model. Default = "cpu".

        bipolar (bool): Boolean flag to enable/disable bipolar input/output
        conversion. Default = False.

    Returns:

        float: Accuracy achieved by model.
    """

    # ensure model is in eval mode
    model.eval()

    # collect outputs to eventually compute accuracy with
    y_target = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            target = target.cpu().float()

            # convert input to bipolar if enabled
            if bipolar:
                inputs = 2 * inputs - 1

            # run inference with model
            output = model(inputs.float())

            if not bipolar:
                # run the output through sigmoid
                output = torch.sigmoid(output)

                # compare against a threshold of 0.5 to generate 0/1 for prediction
                pred = (output.detach().cpu().numpy() > 0.5) * 1
                y_pred.extend(pred.reshape(-1).tolist())
            else:
                y_pred.extend(list(output.flatten().cpu().numpy()))
                target = 2 * target - 1

            y_target.extend(target.tolist())

    return accuracy_score(y_target, y_pred)


def verify(model, brevitas_model, tensors, device: str = "cpu", bipolar: bool = False) -> bool:
    """
    Verifies that the output of a model matches the output from the original
    Brevitas software model.

    Args:

        model: Input model to run inference on.

        brevitas_model: Input Brevitas model to run inference on.

        tensors: Input tensors to run inference with.

        device (str): Device name on which to run the model. Default = "cpu".

        bipolar (bool): Boolean flag to enable/disable bipolar input/output
        conversion. Default = False.

    Returns:

        bool: Pass/fail boolean flag.
    """

    # ensure model is in eval mode
    model.eval()
    brevitas_model.eval()

    verify_pass = True
    ok, nok = 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0

    verify_range = trange(tensors.shape[0], desc="Verification", position=0, leave=True)
    input_vec_len = tensors.shape[1]

    with torch.no_grad():
        for i in verify_range:
            # get inputs to run inference with
            inputs = tensors[i].to(device).reshape((1, input_vec_len))

            # run inference with original brevitas model
            golden_output = brevitas_model(inputs.float())

            # run the output through sigmoid
            golden_output = torch.sigmoid(golden_output)

            # compare against a threshold of 0.5 to generate 0/1 for prediction
            golden_output = (golden_output.detach().cpu().numpy() > 0.5) * 1

            # convert input to bipolar if enabled
            if bipolar:
                inputs = 2 * inputs - 1

            # run inference with model
            output = model(inputs.float())

            if bipolar:
                #  bipolarize output of brevitas model and convert output to
                #  numpy array
                golden_output = 2.0 * golden_output - 1.0
                output = output.detach().cpu().numpy()
            else:
                # run the output through sigmoid
                output = torch.sigmoid(output)

                # compare against a threshold of 0.5 to generate 0/1 for prediction
                output = (output.detach().cpu().numpy() > 0.5) * 1

            # compare the outputs
            ok += 1 if golden_output == output else 0
            nok += 1 if golden_output != output else 0

            if golden_output == output:
                if golden_output == 1:
                    tp += 1
                elif golden_output == -1:
                    tn += 1
                else:
                    print(f"Unexpected Brevitas model output == {golden_output}")
            else:
                verify_pass = False
                if golden_output == 1:
                    fn += 1
                elif golden_output == -1:
                    fp += 1
                else:
                    print(f"Unexpected Brevitas model output == {golden_output}")

            # update the trange description
            verify_range.set_description(
                "ok %d nok %d (tp=%d, tn=%d, fp=%d, fn=%d)" % (ok, nok, tp, tn, fp, fn)
            )
            verify_range.refresh()

    return verify_pass


def verify_onnx(onnx_model, brevitas_model, tensors, device: str = "cpu") -> bool:
    """
    Verifies that the output of an ONNX model matches the output from the
    original Brevitas software model.

    Args:

        onnx_model: Input model to run inference on.

        brevitas_model: Input Brevitas model to run inference on.

        tensors: Input tensors to run inference with.

        device (str): Device name on which to run the model. Default = "cpu".

    Returns:

        bool: Pass/fail boolean flag.
    """

    # ensure model is in eval mode
    brevitas_model.eval()

    verify_pass = True
    ok, nok = 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0

    verify_range = trange(tensors.shape[0], desc="Verification (ONNX)", position=0, leave=True)
    input_vec_len = tensors.shape[1]

    with torch.no_grad():
        for i in verify_range:
            # get inputs to run inference with
            inputs = tensors[i].to(device).reshape((1, input_vec_len))

            # run inference with original brevitas model
            golden_output = brevitas_model(inputs.float())
            golden_output = torch.sigmoid(golden_output)
            golden_output = (golden_output.detach().cpu().numpy() > 0.5) * 1
            golden_output = 2.0 * golden_output - 1.0

            # run inference with onnx model
            finnonnx_in_tensor_name = onnx_model.graph.input[0].name
            finnonnx_model_in_shape = onnx_model.get_tensor_shape(finnonnx_in_tensor_name)
            finnonnx_out_tensor_name = onnx_model.graph.output[0].name
            inputs = inputs.detach().numpy()
            inputs = 2 * inputs - 1
            inputs = inputs.reshape(finnonnx_model_in_shape)
            input_dict = {finnonnx_in_tensor_name: inputs}
            output_dict = oxe.execute_onnx(onnx_model, input_dict)
            output = output_dict[finnonnx_out_tensor_name][0][0]

            # compare the outputs
            ok += 1 if golden_output == output else 0
            nok += 1 if golden_output != output else 0

            if golden_output == output:
                if golden_output == 1:
                    tp += 1
                elif golden_output == -1:
                    tn += 1
                else:
                    print(f"Unexpected Brevitas model output == {golden_output}")
            else:
                verify_pass = False
                if golden_output == 1:
                    fn += 1
                elif golden_output == -1:
                    fp += 1
                else:
                    print(f"Unexpected Brevitas model output == {golden_output}")

            # update the trange description
            verify_range.set_description(
                "ok %d nok %d (tp=%d, tn=%d, fp=%d, fn=%d)" % (ok, nok, tp, tn, fp, fn)
            )
            verify_range.refresh()

    return verify_pass
