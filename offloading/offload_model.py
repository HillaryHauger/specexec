import accelerate

import torch
import transformers
from tqdm.auto import trange

from offloading.model_loader import Loader
from offloading.offload_engine import OffoadingCache
from offloading.storage_wrapper import ModuleWithStorage

from specdec import utils

if "logger" not in globals():
    logger = utils.get_logger()


def load_offloaded_model(
    model_name,
    device_size=3,
    main_device=torch.device("cuda"),
    main_dtype=torch.float16,
):

    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "meta",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
        "model.rotary_emb": "cuda:0",
    }

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # load config
    model_config = transformers.AutoConfig.from_pretrained(model_name)
    # model_config = transformers.LlamaModel._autoset_attn_implementation(model_config)  # fix config._attn_implementation  # IS IT NEEDED W/O GPTQ ?
    # should be 'sdpa' if enabled / available, see model._check_and_enable_sdpa(...) also transformers.utils.is_torch_sdpa_available()

    loader = Loader(model_name)

    def make_module():
        with torch.device(main_device):
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            module = type(model.model.layers[0])(model_config, layer_idx=0)
            torch.set_default_dtype(original_dtype)

        # module = deepcopy(sample_layer)
        module.layer_idx = None
        return ModuleWithStorage(module.to(device=main_device, dtype=main_dtype))

    with utils.Timing(synchronize=True) as t:
        cache = OffoadingCache(
            make_module, device_size=device_size
        )  # <-- keep :device_size: modules on device
    logger.debug(f"OffoadingCache created in {t.elapsed:.6f}")

    # offsets, storage_size_bytes = cache.get_storage_offsets()
    offsets = cache.all_device_buffers[0].offsets
    logger.debug("offsets, storage_size_bytes", offsets)

    for layer_idx in trange(
        model.config.num_hidden_layers, desc="Filling CPU modules storage"
    ):
        logger.debug(f"start loading layer {layer_idx} {'.' * 48}")
        with accelerate.init_empty_weights():
            layer_ = type(model.model.layers[0])(model_config, layer_idx)
        logger.debug(f"created empty layer {layer_idx}")

        loader.fill_layer(layer_, layer_idx)
        logger.debug(f"filled layer {layer_idx}")

        module = ModuleWithStorage(layer_.to(dtype=main_dtype), offsets=offsets)
        logger.debug(f"MWS layer {layer_idx}")

        cache.add_module(uid=layer_idx, module=module)
        logger.debug(f"added layer {layer_idx}")

        del layer_
        logger.debug(f"done with layer {layer_idx}")

    model.model.layers = OffloadedLayerIter(cache)

    return model


class OffloadedLayerIter(torch.nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache  # this module owns cache

    def __iter__(self):
        for layer_idx, module in self.cache.load_modules(
            *range(len(self.cache.offloaded_storages))
        ):
            module.module.self_attn.layer_idx = layer_idx
            yield module

    def __getitem__(self, idx):
        return self.cache.all_device_buffers[idx]
