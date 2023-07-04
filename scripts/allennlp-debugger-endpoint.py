import argparse
import shutil
import sys

from allennlp.commands import main

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str)
parser.add_argument('--overrides', type=str, required=False)  # , default=json.dumps({"trainer": {"cuda_device": -1}}))
parser.add_argument('--serialization', type=str, required=False)
parser.add_argument('--include-package', type=str)
parser.add_argument('--recover', action='store_true')
ARGS = parser.parse_args()

config_file = ARGS.config_file
# Use overrides to train on CPU.
overrides = ARGS.overrides

serialization_dir = ARGS.serialization if ARGS.serialization is not None else '$TMP/debugger_train'

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
if not ARGS.recover:
    shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", ARGS.include_package,
    "-o", overrides if overrides is not None else "{}"
]

if ARGS.recover:
    sys.argv.append('--recover')

main()
