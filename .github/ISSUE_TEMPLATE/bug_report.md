---
name: Bug report
about: Something isn't working as expected
title: ''
labels: bug
assignees: ''

---

## Prerequisites
Please make sure to check off these prerequisites before submitting a bug report.
- [ ] Test that the bug appears on the current version of the dev-branch. Make sure to include the commit hash of the commit you checked out.
- [ ] Check that the issue hasn't already been reported, by checking the currently open issues.
- [ ] If there are steps to reproduce the problem, make sure to write them down below.
- [ ] If relevant, please include the ONNX files, which were created directly before and/or after the bug.

## Quick summary
Please give a brief and concise description of the bug.

## Details
Please add to the following sections to describe the bug as accurately as possible.

### Steps to Reproduce
Add what needs to be done to reproduce the bug. Add code examples where useful. For build scripts, [saving](https://finn-dev.readthedocs.io/en/latest/source_code/finn.builder.html#finn.builder.build_dataflow_config.DataflowBuildConfig.save_intermediate_models) the intermediate (and resulting) ONNX models and including them here is extremely useful.

#### For example notebooks:
1. Set up the [...] board using the following image: [...]
2. Install the finn-examples package with the following commands: [...]
3. Run notebook [...] on the [...] board with the following commands: [...]
4. [Further steps ...]

#### For build scripts:
1. Clone the FINN-Examples repository
2. Checkout the dev branch, with commit hash: [...]
3. Start the docker container with the command: [...]
4. Run the build script [...] with the following commands: [...]
5. [Further steps ...]

### Expected behavior
Please add a brief description of what you expected to happen.

### Actual behavior
Describe what actually happens instead.

## Optional

### Possible fix
If you already know where the issue stems from, or you have a hint please let us know.

### Additional context
Add any other context about the problem here.
