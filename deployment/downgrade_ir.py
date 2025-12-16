import onnx
import sys

if len(sys.argv) < 2:
    print("Usage: python downgrade_ir.py <model.onnx>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Loading {model_path}...")
model = onnx.load(model_path)

print(f"Current IR version: {model.ir_version}")
if model.ir_version > 9:
    print("Downgrading IR version to 9...")
    model.ir_version = 9
    onnx.save(model, model_path)
    print(f"Saved {model_path} with IR version 9.")
else:
    print("IR version is already <= 9.")



