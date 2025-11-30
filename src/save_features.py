import argparse
import os
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple

import torch

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel


@torch.no_grad()
def inference(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_function: Callable,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:

    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    start_time = time.time()
    is_batched = args.batch_size > 1
    for i, item in enumerate(tqdm(loader, desc=f"Running inference with batch size {args.batch_size} for split {args.split}")):
        text = item["text"] if is_batched else item["text"][0]
        image_path = item["image"] if is_batched else item["image"][0]
        # prompt is a concat of text (input) + response (output). 
        # since we're generating we don't supply the output, so resp should be empty.
        resp = [""] * len(text) if is_batched else ""
        # print(f"INFERENCE_TYPE: is_batched={is_batched} text={type(text)}, path={type(image_path)}")
        inputs = model_class.preprocessor(
            instruction=text,
            image_file=image_path,
            response=resp,
            generation_mode=args.generation_mode,
        )

        # inputs.keys(): dict_keys(['input_ids', 'attention_mask', 'pixel_values'])

        # do_sample False = greedy decoding; faster; deterministic
        if args.generation_mode:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        else:
            out = model(**inputs).logits

        print(out.shape)
        # TODO: look at hooks and see if they need to be rework
        item["model_output"] = out
        # output consists of prompt tokens followed by generated caption tokens
        # calculate input length to find where in the generated sequence the output begins
        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
        # output in terms of token IDs
        item["model_generated_output"] = out[:, input_len:]
        # output in decoded text
        item["model_predictions"] = model_class.get_tokenizer().batch_decode(
            out[:, input_len:], skip_special_tokens=True
        )

        # TODO: rough batching fix: break batch into loops in here
        if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**item)
                    if hook_output:
                        item.update(hook_output)

        hook_data = update_dict_of_list(item, hook_data)
        clear_hooks_variables()
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
    return hook_data


if __name__ == "__main__":

    args = get_arguments()
    if args.batch_size <= 0:
        raise ValueError("Invalid batch size")

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))

    set_seed(args.seed)

    logger.info(f"Loading model: {args.model_name_or_path}")
    log_args(args, logger)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

    # logger.info(f"Hook names: {args.hook_names}")
    # > save_hidden_states_for_token_of_interest
    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args,
    )

    hook_data = inference(
        loader=loader,
        model_class=model_class,
        device=device,
        hook_return_function=hook_return_functions,
        logger=logger,
        args=args,
    )

    clear_forward_hooks(model_class.model_)
    # TODO: rough batching fix: break batch into loops in here
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, args=args, logger=logger)
    
    
    # hook_return_function = register_hooks() =
    # get_hidden_states(
    #   extract_token_of_interest=True,
    #   token_of_interest_idx=token_of_interest_idx,
    #   token_of_interest_start_token=args.token_of_interest_start_token,
    #   save_only_generated_tokens=args.save_only_generated_tokens,
    #   data=hook_data, args=args,logger=logger
    # )

    # hook_postprocessing_function = hooks_postprocessing() = 
    # save_hidden_states_to_file(
    #   args=args,
    #   data_keys=data_keys,
    #   hook_name=hook_name,
    #   data=hook_data, args=args,logger=logger
    # )

    # ==> get_hidden_states() run on each iteration; save_hidden_states_to_file() run after all iterations
    # TODO: check both these states for updating w/ batching behavior