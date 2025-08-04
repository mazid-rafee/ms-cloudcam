import os
import random
import numpy as np
import torch
from utils.trainer_tester import evaluate_test

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate_and_log(model, model_path, test_loader, device, log_file, description):
    print(f"\nEvaluating {description} model:")
    model.load_state_dict(torch.load(model_path))
    results = evaluate_test(model, test_loader, device)
    print("".join(results))
    log_file.write(f"\nEvaluation of {description} Model:\n")
    log_file.writelines(results)
    log_file.write("\n")
