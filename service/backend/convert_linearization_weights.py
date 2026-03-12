#!/usr/bin/env python3
"""Convert SingleHDR TF checkpoints to PyTorch (dequantization + linearization only).

Downloads checkpoints from Google Drive, converts to .pt, and downloads invemor.txt.
This is a subset of singlehdr-service's convert_weights.py, keeping only what
IntrinsicHDR needs for neural linearization.

Usage:
    python convert_linearization_weights.py \
        --ckpt_deq /path/to/ckpt_deq/model.ckpt \
        --ckpt_lin /path/to/ckpt_lin/model.ckpt \
        --output_dir /path/to/weights/
"""

import argparse
import os
import sys

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    print("ERROR: TensorFlow is required for conversion. Install with: pip install tensorflow-cpu")
    sys.exit(1)


def load_tf_vars(ckpt_path):
    """Load all variables from a TF checkpoint as a dict {name: numpy_array}."""
    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = reader.get_variable_to_shape_map()
    result = {}
    for name in sorted(var_map.keys()):
        if 'Adam' in name or 'global_step' in name or 'beta1_power' in name or 'beta2_power' in name:
            continue
        result[name] = reader.get_tensor(name)
    return result


def conv_w(tf_w):
    """TF conv weight [H,W,Cin,Cout] -> PyTorch [Cout,Cin,H,W]."""
    return torch.from_numpy(tf_w.transpose(3, 2, 0, 1).copy())


def fc_w(tf_w):
    """TF dense weight [in,out] -> PyTorch [out,in]."""
    return torch.from_numpy(tf_w.T.copy())


def to_tensor(arr):
    return torch.from_numpy(arr.copy())


def convert_dequantization(tf_vars, prefix="Dequantization_Net"):
    pt_names = [
        'conv_in1', 'conv_in2',
        'down1_c1', 'down1_c2', 'down2_c1', 'down2_c2',
        'down3_c1', 'down3_c2', 'down4_c1', 'down4_c2',
        'up1_c1', 'up1_c2', 'up2_c1', 'up2_c2',
        'up3_c1', 'up3_c2', 'up4_c1', 'up4_c2',
        'conv_out',
    ]
    state_dict = {}
    for i, pt_name in enumerate(pt_names):
        tf_suffix = f"conv2d{'_' + str(i) if i > 0 else ''}"
        k_name = f"{prefix}/{tf_suffix}/kernel"
        b_name = f"{prefix}/{tf_suffix}/bias"
        state_dict[f"{pt_name}.weight"] = conv_w(tf_vars[k_name])
        state_dict[f"{pt_name}.bias"] = to_tensor(tf_vars[b_name])
    return state_dict


def _map_conv_bn(state_dict, tf_vars, tf_prefix, tf_conv_name, pt_prefix, has_bias=True):
    k = f"{tf_prefix}/{tf_conv_name}/weights"
    state_dict[f"{pt_prefix}.conv.weight"] = conv_w(tf_vars[k])
    if has_bias:
        b = f"{tf_prefix}/{tf_conv_name}/biases"
        if b in tf_vars:
            state_dict[f"{pt_prefix}.conv.bias"] = to_tensor(tf_vars[b])

    if tf_conv_name == 'conv1':
        bn_prefix = f"{tf_prefix}/bn_conv1"
    else:
        bn_prefix = f"{tf_prefix}/bn{tf_conv_name[3:]}"

    bn_base = f"{bn_prefix}/BatchNorm"
    if f"{bn_base}/gamma" in tf_vars:
        state_dict[f"{pt_prefix}.bn.weight"] = to_tensor(tf_vars[f"{bn_base}/gamma"])
        state_dict[f"{pt_prefix}.bn.bias"] = to_tensor(tf_vars[f"{bn_base}/beta"])
        state_dict[f"{pt_prefix}.bn.running_mean"] = to_tensor(tf_vars[f"{bn_base}/moving_mean"])
        state_dict[f"{pt_prefix}.bn.running_var"] = to_tensor(tf_vars[f"{bn_base}/moving_variance"])
        state_dict[f"{pt_prefix}.bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)


def convert_crf_feature_net(tf_vars, prefix="crf_feature_net"):
    state_dict = {}
    _map_conv_bn(state_dict, tf_vars, prefix, 'conv1', 'crf_feature_net.conv1', has_bias=True)

    res_blocks = [
        ('res2a', True), ('res2b', False), ('res2c', False),
        ('res3a', True), ('res3b', False),
    ]
    for block_name, has_branch1 in res_blocks:
        if has_branch1:
            _map_conv_bn(state_dict, tf_vars, prefix,
                         f'{block_name}_branch1', f'crf_feature_net.{block_name}_b1',
                         has_bias=False)
        for sub in ['branch2a', 'branch2b', 'branch2c']:
            _map_conv_bn(state_dict, tf_vars, prefix,
                         f'{block_name}_{sub}', f'crf_feature_net.{block_name}_b2{sub[-1]}',
                         has_bias=False)
    return state_dict


def convert_ae_invcrf(tf_vars, prefix="ae_invcrf_decode_net"):
    state_dict = {}
    state_dict["ae_invcrf_decode_net.fc.weight"] = fc_w(tf_vars[f"{prefix}/dense/kernel"])
    state_dict["ae_invcrf_decode_net.fc.bias"] = to_tensor(tf_vars[f"{prefix}/dense/bias"])
    return state_dict


def convert(ckpt_deq, ckpt_lin, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dequantization checkpoint...")
    tf_deq = load_tf_vars(ckpt_deq)
    deq_sd = convert_dequantization(tf_deq)
    torch.save(deq_sd, os.path.join(output_dir, "dequantization.pt"))
    print(f"  Saved {len(deq_sd)} parameters")

    print("Loading linearization checkpoint...")
    tf_lin = load_tf_vars(ckpt_lin)
    lin_sd = {}
    lin_sd.update(convert_crf_feature_net(tf_lin))
    lin_sd.update(convert_ae_invcrf(tf_lin))
    torch.save(lin_sd, os.path.join(output_dir, "linearization.pt"))
    print(f"  Saved {len(lin_sd)} parameters")

    print(f"\nLinearization weights saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert SingleHDR TF checkpoints to PyTorch (linearization only)")
    parser.add_argument("--ckpt_deq", required=True, help="Dequantization checkpoint path")
    parser.add_argument("--ckpt_lin", required=True, help="Linearization checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Output directory for .pt files")
    args = parser.parse_args()
    convert(args.ckpt_deq, args.ckpt_lin, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
