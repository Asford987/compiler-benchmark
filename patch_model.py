#!/usr/bin/env python3
"""
flatten_for_compilednn.py
Usage:  python3 flatten_for_compilednn.py input.h5 output_flat.h5
"""

import sys
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# ----------------------------------------------------------------------
def stitch_layers(src_model, x):
    """
    Recursively apply layers of `src_model` to tensor x, flattening
    any nested Sequential/Functional blocks.
    """
    for layer in src_model.layers:
        # If the layer is itself a model (Sequential or Functional),
        # dive into its sub-layers.
        if isinstance(layer, tf.keras.Model):
            x = stitch_layers(layer, x)
        else:
            x = layer(x)
    return x
# ----------------------------------------------------------------------

if len(sys.argv) != 3:
    print("Usage: python flatten_for_compilednn.py  in_model.h5  out_flat.h5")
    sys.exit(1)

in_path, out_path = sys.argv[1:3]

orig = load_model(in_path, compile=False)

# True data-tensor shape  (None, H, W, C) or (None, C, H, W) …
shape_with_batch = orig.input_shape
if isinstance(shape_with_batch, list):
    shape_with_batch = shape_with_batch[0]
data_shape = shape_with_batch[1:]          # drop the None

# Build flat functional graph
inp  = Input(shape=data_shape, name="compilednn_input")
out  = stitch_layers(orig, inp)
flat = Model(inp, out, name="flattened_for_compilednn")

# Copy weights (order preserved because we walked the same sequence)
flat.set_weights(orig.get_weights())

flat.save(out_path)
print(f"✅  Saved flat, batch-less model to  {out_path}")
print(f"    Final input shape: {flat.input_shape}")
