from pathlib import Path
import shutil
import os
import glob
from multiprocessing import Manager
import re

def create_clean_dir(dir_name:Path, force:bool = True):
    if dir_name.exists() and force:
        remove_contents_in_dir(dir_name)
    else:
        dir_name.mkdir(exist_ok=True, parents=True)


def copy_data_dir(dataset_path, output_path):
    # Copy utt2spk wav.scp and so on, but not the directories inside (may contains clear or anonymzied *.wav)
    os.makedirs(output_path, exist_ok=True)
    for p in glob.glob(str(dataset_path / '*'), recursive=False):
        if os.path.isfile(p):
            shutil.copy(p, output_path)


def remove_contents_in_dir(dir_name:Path):
    # solution from https://stackoverflow.com/a/56151260
    for path in dir_name.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)



# def scan_checkpoint(cp_dir, prefix):
#     pattern = os.path.join(cp_dir, prefix + '*****')
#     cp_list = glob.glob(pattern)
#     if len(cp_list) == 0:
#         return None

#     try:
#         cp_list_by_name = sorted([(int(ckpt.split(prefix)[-1]), ckpt) for ckpt in cp_list])
#         return Path(cp_list_by_name[-1][1])
#     except ValueError:
#         # Handle the case where conversion to int fails
#         return None
def _parse_train_log_for_best_epoch(cp_dir):
    """Parse train_log.txt to find epoch with lowest valid loss.
    Format: 'epoch: N, lr: ... - train loss: X - valid loss: Y, valid ErrorRate: Z'
    Returns (best_epoch, min_valid_loss) or (None, None) if not found."""
    log_path = Path(cp_dir) / "train_log.txt"
    if not log_path.exists():
        return None, None
    best_epoch, min_loss = None, float("inf")
    # epoch: 1, lr: 1.44e-04 - train loss: 14.85 - valid loss: 14.58, valid ErrorRate: 9.99e-01
    pat = re.compile(r"epoch:\s*(\d+).*?valid loss:\s*([\d.e+-]+)", re.DOTALL)
    try:
        with open(log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    ep, loss = int(m.group(1)), float(m.group(2))
                    if loss < min_loss:
                        min_loss, best_epoch = loss, ep
    except Exception:
        return None, None
    return best_epoch, min_loss if best_epoch is not None else None


def scan_checkpoint(cp_dir, prefix, select_by_valid_loss=True):
    """Select checkpoint: by lowest valid loss (from train_log.txt) if select_by_valid_loss,
    else by max numeric suffix (epoch)."""
    cp_dir = Path(cp_dir)
    pattern = str(cp_dir / (prefix + '*'))
    cp_list = glob.glob(pattern)

    if select_by_valid_loss:
        best_epoch, _ = _parse_train_log_for_best_epoch(cp_dir)
        if best_epoch is not None:
            # Checkpoint names: CKPT+1, CKPT+2 (SpeechBrain format)
            ckpt_name = f"{prefix}+{best_epoch}" if not prefix.endswith("+") else f"{prefix}{best_epoch}"
            ckpt_path = cp_dir / ckpt_name
            if ckpt_path.exists() and ckpt_path.is_dir():
                return ckpt_path

    numeric_ckpts = []
    for ckpt in cp_list:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_dir():
            continue
        name = ckpt_path.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if re.fullmatch(r"\+?\d+", suffix):
            numeric = int(suffix.lstrip("+"))
            numeric_ckpts.append((numeric, ckpt_path))

    if numeric_ckpts:
        return max(numeric_ckpts, key=lambda x: x[0])[1]
    return None


def get_datasets(config):
    datasets = {}
    data_dir = config.get('data_dir', None).expanduser() # if '~' is given in path then manually expand
    for dataset in config['datasets']:
        no_sub = True
        for subset in ['trials', 'enrolls']:
            if subset in dataset:
                for subset in dataset[subset]:
                    dataset_name = f'{dataset["data"]}{subset}'
                    datasets[dataset_name] = Path(data_dir, dataset_name)
                    no_sub = False
        if no_sub:
            dataset_name = f'{dataset["data"]}'
            datasets[dataset_name] = Path(data_dir, dataset_name)
    return datasets
