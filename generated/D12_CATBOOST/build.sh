#!/bin/bash

# Environment Setup
export SDE=$HOME/bf-sde-9.13.4
export SDE_INSTALL=$SDE/install
export PATH=$SDE_INSTALL/bin:$PATH

# Configuration
P4_PROGRAM_NAME="d12"
P4_SOURCE="d12.p4"
BUILD_DIR="build"

echo "=========================================="
echo "Compiling $P4_PROGRAM_NAME"
echo "=========================================="

mkdir -p $BUILD_DIR

bf-p4c --target tofino \
       --arch tna \
       --create-graphs \
       --verbose 2 \
       --display-power-budget \
       --parser-timing-reports \
       --program-name $P4_PROGRAM_NAME \
       -o $BUILD_DIR \
       $P4_SOURCE

if [ $? -eq 0 ]; then
    ABS_BUILD="$PWD/$BUILD_DIR"
    python3 - "$BUILD_DIR/$P4_PROGRAM_NAME.conf" "$ABS_BUILD" <<'PYEOF'
import sys, json
conf_path, abs_build = sys.argv[1], sys.argv[2]
with open(conf_path) as f: conf = json.load(f)
for dev in conf.get('p4_devices', []):
    for prog in dev.get('p4_programs', []):
        prog.pop('model_json_path', None)
        prog['bfrt-config'] = f"{abs_build}/bfrt.json"
        for pipe in prog.get('p4_pipelines', []):
            pipe['context'] = f"{abs_build}/{pipe['p4_pipeline_name']}/context.json"
            pipe['config']  = f"{abs_build}/{pipe['p4_pipeline_name']}/tofino.bin"
            pipe['path']    = abs_build
with open(conf_path, 'w') as f: json.dump(conf, f, indent=4)
PYEOF
    cp "$BUILD_DIR/$P4_PROGRAM_NAME.conf" "./$P4_PROGRAM_NAME.conf"
    echo "------------------------------------------"
    echo "SUCCESS: Compilation finished."
    echo "Output directory: $PWD/$BUILD_DIR"
    echo "------------------------------------------"
else
    echo "------------------------------------------"
    echo "ERROR: Compilation failed."
    echo "------------------------------------------"
    exit 1
fi
