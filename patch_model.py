import zipfile
import json
import os
import shutil

keras_file = "MobileNetV2 Waste Management.keras"
fixed_file = "MobileNetV2 Waste Management_fixed.keras"
extract_dir = "/tmp/keras_extract_new"

# Clean up any previous extraction
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)

# Extract the existing zip
with zipfile.ZipFile(keras_file, 'r') as z:
    z.extractall(extract_dir)
    all_names = z.namelist()
    config_data = z.read("config.json")
    metadata = z.read("metadata.json")

config = json.loads(config_data.decode('utf-8'))

# Recursively remove ALL occurrences of quantization_config
def remove_quant(obj):
    if isinstance(obj, dict):
        obj.pop('quantization_config', None)
        for v in obj.values():
            remove_quant(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_quant(item)

remove_quant(config)

# Write patched zip
with zipfile.ZipFile(fixed_file, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("config.json", json.dumps(config))
    z.writestr("metadata.json", metadata)
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file in ("config.json", "metadata.json"):
                continue
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, extract_dir)
            z.write(abs_path, rel_path)

# Replace original
os.replace(fixed_file, keras_file)
print("✅ Model patched successfully — quantization_config removed from all layers.")
