import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F

from config import Config
from model import Generator, LayerwiseDecoder
from train import find_latest_checkpoint
from data import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description="Test Generator→LayerwiseDecoder round-trip on a dataset sample.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory from config")
    parser.add_argument("--module", type=int, default=None, help="Module index to run decoder on (default: the module that owns a decoder)")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of sequences to test")
    parser.add_argument("--skip-docs", type=int, default=0, help="Skip N documents before sampling")
    parser.add_argument("--layer", type=int, default=None, help="Layer to show text reconstructions for (default: last)")
    parser.add_argument("--show-chars", type=int, default=300, help="Characters to show in sample output")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device)

    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir

    ckpt_path = args.checkpoint or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        print("No checkpoint found.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("cfg", cfg)

    # New checkpoint format: a single file holding a list of per-module state_dicts.
    modules = ckpt.get("modules")
    if not modules:
        print("No 'modules' list in this checkpoint.")
        return

    # Only the decoder-owning module (module 0) carries a layerwise_decoder, and that decoder is
    # trained to reconstruct from *that* module's latents — so default the probe to it.
    decoder_modules = [i for i, m in enumerate(modules) if "layerwise_decoder" in m]
    if args.module is not None:
        module_idx = args.module
    elif decoder_modules:
        module_idx = decoder_modules[0]
    else:
        print("No module in this checkpoint owns a layerwise_decoder.")
        return
    if not (0 <= module_idx < len(modules)):
        print(f"--module {module_idx} out of range (checkpoint has {len(modules)} modules, 0..{len(modules)-1})")
        return

    print(f"Loading module {module_idx} from {ckpt_path}\n")

    # Load all previous frozen modules to compute prev_latent
    prev_generators = []
    for i in range(module_idx):
        prev_gen = Generator(ckpt_cfg, layer_idx=i).to(device)
        prev_gen.load_state_dict(modules[i]["generator"], strict=False)
        prev_gen.eval()
        for p in prev_gen.parameters():
            p.requires_grad_(False)
        prev_generators.append(prev_gen)
        print(f"  Loaded frozen module {i}")

    final_state = modules[module_idx]
    final_generator = Generator(ckpt_cfg, layer_idx=module_idx).to(device)
    final_generator.load_state_dict(final_state["generator"], strict=False)
    final_generator.eval()

    if "layerwise_decoder" not in final_state:
        print("No layerwise_decoder in this module's checkpoint.")
        return
    layerwise_decoder = LayerwiseDecoder(ckpt_cfg).to(device)
    layerwise_decoder.load_state_dict(final_state["layerwise_decoder"])
    layerwise_decoder.eval()

    n_layers = ckpt_cfg.n_layers
    show_layer = args.layer if args.layer is not None else n_layers - 1
    if not (0 <= show_layer < n_layers):
        print(f"--layer {show_layer} out of range (model has {n_layers} layers, 0-indexed 0..{n_layers-1})")
        return

    tokenizer = ByteTokenizer()
    T = ckpt_cfg.context_length

    from datasets import load_dataset as _hf_load
    stream = _hf_load("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    stream = stream.skip(args.skip_docs)

    samples = []
    for doc in stream:
        raw = tokenizer.encode(doc["text"])
        if len(raw) >= T:
            samples.append(torch.tensor(raw[:T], dtype=torch.long))
        if len(samples) >= args.n_samples:
            break

    if not samples:
        print("No documents long enough found.")
        return

    batch = torch.stack(samples).to(device)  # [N, T]
    # Match training: x = batch[:, :-1], target = x
    x = batch[:, :-1]  # [N, T-1]

    with torch.no_grad():
        # Run through all previous frozen modules to get prev_latent
        prev_latent = None
        for gen in prev_generators:
            prev_latent = gen.encode_clean(x, prev_latent)

        # Returns [h_0, h_1, ..., h_{n_layers}], length = n_layers + 1
        hiddens = final_generator.forward_hidden_layerwise(x, prev_latent)

        print(f"Samples  : {len(samples)}")
        print(f"Modules  : {module_idx + 1}  (running decoder on module {module_idx})")
        print(f"Layers   : {n_layers}")
        print(f"Seq len  : {x.shape[1]}")
        print()

        pred_ids_per_layer = []
        for l in range(n_layers):
            h = hiddens[l + 1]                                        # [N, T-1, D]
            logits = layerwise_decoder(l, h)                          # [N, T-1, vocab_size]
            ce = F.cross_entropy(logits.reshape(-1, ckpt_cfg.vocab_size), x.reshape(-1)).item()
            pred_ids = logits.argmax(dim=-1)                          # [N, T-1]
            acc = (pred_ids == x).float().mean().item()
            pred_ids_per_layer.append(pred_ids)
            print(f"Layer {l:2d}: CE={ce:.4f}  Acc={acc*100:.2f}%")

    print("=" * 70)
    print(f"\nReconstructions (layer {show_layer}):")
    print()

    pred_ids = pred_ids_per_layer[show_layer]
    for i in range(len(samples)):
        orig_ids = x[i].cpu()
        rec_ids = pred_ids[i].cpu()
        s_acc = (rec_ids == orig_ids).float().mean().item()
        orig = tokenizer.decode(orig_ids.tolist())
        recon = tokenizer.decode(rec_ids.tolist())
        n = args.show_chars
        print(f"--- Sample {i+1}  (layer={show_layer}, acc={s_acc*100:.1f}%) ---")
        print(f"ORIG : {repr(orig[:n].replace(chr(10), '↵'))}")
        print(f"RECON: {repr(recon[:n].replace(chr(10), '↵'))}")
        print()


if __name__ == "__main__":
    main()
