from typing import List
from attention_regulation.attn_processor import AttnProcessorAttnMapEdit


class OptimiseAttnMapEditScheme:
    def __init__(
        self,
        target_ids: list,
        replace_id: int,
        internals: dict = None,
        attn_layers_to_edit: List[str] = ["all"],
        num_edit_steps: int = 25,
    ):
        self.attn_layers_to_edit = attn_layers_to_edit
        self.edit_all_layers = True if attn_layers_to_edit[0] == "all" else False
        self.target_ids = target_ids
        self.replace_id = replace_id
        self.internals = internals
        self.current_step = 0
        self.num_edit_steps = num_edit_steps

    def add_hook(self, unet):
        hook_list = []
        for name, module in unet.named_modules():
            name_last_word = name.split(".")[-1]
            if name_last_word == "attn2":
                apply_edit = False
                for layers in self.attn_layers_to_edit:
                    if layers in name:
                        apply_edit = True
                        break
                if apply_edit or self.edit_all_layers:
                    hook_fn = self._get_edit_hook_fn()
                    hook = module.register_forward_hook(hook_fn, with_kwargs=True)
                    hook_list.append(hook)
        return hook_list

    def _get_edit_hook_fn(self):
        attn_processor = AttnProcessorAttnMapEdit()

        def hook_fn(module, args, kwargs, output):
            kwargs["target_ids"] = self.target_ids
            kwargs["replace_id"] = self.replace_id
            kwargs["internals"] = self.internals
            kwargs["current_step"] = self.current_step
            kwargs["num_edit_steps"] = self.num_edit_steps
            modified_output = attn_processor(module, args[0], **kwargs)
            return modified_output

        return hook_fn
