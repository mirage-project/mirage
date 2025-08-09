# NVIDIA Nsight Compute Kernel Profiling Guide

## Prerequisites

### Install NVIDIA Nsight Compute

Download and install NVIDIA Nsight Compute from the official NVIDIA website.

### Environment Setup

Follow the instructions in `README.md` to configure your environment.

> **Warning**: This script is written specifically for `venv`. Please use `venv` as your virtual environment for optimal functionality.

## Getting Started

### Create Your Demo Application

Write a demo application that calls the target kernel you want to profile.

### Setup Profiling Environment

Ensure you are always working from the `./mirage` directory:

```bash
cd ./mirage
```

#### Required Permissions and Directories

Get root permission on your cluster and create the necessary directories:

```bash
mkdir ncu_tmp
mkdir report
```

#### Make Script Executable

```bash
chmod +x analyze.sh
```

### Run the Profiling

Execute the analysis script with your parameters:

```bash
./analyze.sh report/<report-name>.ncu-rep <kernel-name> python demo.py --args
```

> **Warning**: This script can only profile global functions, not device functions.

**Parameters:**
- `<report-name>.ncu-rep`: Name of the output report file
- `<kernel-name>`: Name of the kernel to profile
- `python demo.py --args`: Your application command with arguments

### Transfer and Analyze Results

#### Transfer Report to Local Machine

Use `scp` or other file transfer tools to copy the generated report to your local device:

```bash
scp user@remote:/path/to/mirage/report/<report-name>.ncu-rep ./local/path/
```

#### Open Report in NVIDIA Nsight Compute

Launch NVIDIA Nsight Compute on your local machine and open the transferred `.ncu-rep` file.

### Analyzing Bank Conflicts

For bank conflict analysis, navigate to:
**Details → Memory Workload Analysis**

In this section, you'll find comprehensive metrics related to memory bank conflicts and other memory performance characteristics.

## Troubleshooting

- Ensure you have proper CUDA toolkit installation
- Verify that your target application runs successfully before profiling
- Check that all required directories have proper permissions
- Make sure your virtual environment is properly activated