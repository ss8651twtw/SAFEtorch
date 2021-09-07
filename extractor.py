from utils.function_normalizer import FunctionNormalizer
from utils.instructions_converter import InstructionsConverter
from utils.capstone_disassembler import disassemble
from utils.radare_analyzer import BinaryAnalyzer
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
import torch
import json
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize SAFE
config = Config()
safe = SAFE(config).to(device)

# load instruction converter and normalizer
I2V_FILENAME = "model/word2id.json"
converter = InstructionsConverter(I2V_FILENAME)
normalizer = FunctionNormalizer(max_instruction=150)

# load SAFE weights
SAFE_torch_model_path = "model/SAFEtorch.pt"
state_dict = torch.load(SAFE_torch_model_path)
safe.load_state_dict(state_dict)
safe = safe.eval()

# analyze the binary
binary = BinaryAnalyzer(sys.argv[1])
offsets = binary.get_functions()
off2name = binary.get_function_names(offsets)

def safe_vec(addr):
    asm = binary.get_hexasm(addr)
    arch = binary.arch
    bits = binary.bits

    instructions = disassemble(asm, arch, bits)

    converted_instructions = converter.convert_to_ids(instructions)
    instructions, length = normalizer.normalize_functions([converted_instructions])
    length = torch.LongTensor(length).to(device)
    tensor = torch.LongTensor(instructions[0]).to(device)
    function_embedding = safe(tensor, length, device)

    return function_embedding[0].tolist()

nodes, edges = [], []

for off in offsets:
    name = off2name[off]
    vec = safe_vec(off)
    nodes.append({"offset": off, "name": name, "vector": vec})

for f in binary.afl:
    off = f["offset"]
    idx = offsets.index(off)
    to_idx = set()
    for call in f.get("callrefs", []):
        if call["type"] == "CALL":
            to_idx.add(offsets.index(call["addr"]))
    for to in to_idx:
        edges.append([idx, to])

print(json.dumps({"nodes": nodes, "edges": edges}))
