import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple
import copy

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables, get_most_free_gpu,
                           get_vocab_idx_of_target_token, get_first_pos_of_token_of_interest,
                           set_seed, setup_hooks, update_dict_of_list)
from models import get_model_class, get_module_by_path
from models.image_text_model import ImageTextModel

from save_features import inference
from analysis import analyse_features
from analysis.feature_decomposition import get_feature_matrix, project_representations


# WIP: dictionary learning for concept