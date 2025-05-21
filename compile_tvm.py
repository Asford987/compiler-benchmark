import tensorflow as tf
from tensorflow import keras
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# Load the .h5 Keras model
model = keras.models.load_model("model.h5")

# Convert to Relay
shape_dict = {"input_1": (1, 28, 28, 1)}  # Adjust input shape
mod, params = relay.frontend.from_keras(model, shape_dict)

# Compile to LLVM (CPU)
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save compiled model
lib.export_library("model.so")
with open("model.json", "w") as f_json:
    f_json.write(lib.get_graph_json())
with open("model.params", "wb") as f_params:
    f_params.write(relay.save_param_dict(lib.get_params()))
