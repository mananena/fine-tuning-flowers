"""python -m experiments.train --help; запускать из корня проекта."""
import argparse
import csv
from dataclasses import dataclass, asdict, fields
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import timm
from timm.data import resolve_model_data_config
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.preprocessing import preprocess

BASE = Path(__file__).resolve().parent

@dataclass
class TrainConfig:
    model: str = 'resnet18.a1_in1k'
    lr: float = 0.001
    backbone_lr: float = 0.0001
    batch_size: int = 16
    epochs: int = 6
    freeze_epochs: int = 2
    weight_decay: float = 0.0001
    seed: int = 42
    size: int = 160
    device: str = 'auto'
    data: str = str(BASE / 'data/raw')
    runs: str = str(BASE / 'runs')
    run_name: str = 'baseline'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def split_data(folder, seed, destination):
    """70/15/15 по классам. Точные дубликаты запрещены, сплит фиксируется."""
    folder = Path(folder)
    classes = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []
    if len(classes) < 3:
        raise ValueError('Нужны минимум 3 папки классов в experiments/data/raw.')
    result = {'classes': classes, 'seed': seed, 'train': [], 'val': [], 'test': []}
    seen = {}
    rng = random.Random(seed)
    for label, name in enumerate(classes):
        paths = sorted(p for p in (folder / name).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
        if len(paths) < 30:
            raise ValueError(f'{name}: {len(paths)} изображений, требуется минимум 30.')
        entries = []
        for p in paths:
            with Image.open(p) as im:
                im.verify()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f'Дубликат: {p} и {seen[digest]}')
            seen[digest] = str(p)
            entries.append({'path': p.relative_to(folder).as_posix(), 'label': label, 'sha256': digest})
        rng.shuffle(entries)
        n_val = max(1, round(len(entries) * .15))
        n_test = max(1, round(len(entries) * .15))
        result['test'].extend(entries[:n_test])
        result['val'].extend(entries[n_test:n_test+n_val])
        result['train'].extend(entries[n_test+n_val:])
    if destination.exists():
        old = read_json(destination)
        if old != result:
            raise ValueError('Данные или seed изменились. Для нового эксперимента создайте новую папку runs.')
    else:
        write_json(destination, result)
    return result

class Photos(Dataset):
    def __init__(self, folder, rows, spec, train=False):
        self.folder, self.rows, self.spec = Path(folder), rows, spec
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(spec['size'], scale=(.75, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(spec['mean'], spec['std'])
        ]) if train else None
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(self.folder / row['path']) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            x = self.augment(im) if self.augment else torch.from_numpy(preprocess(im, self.spec))
        return x, row['label']

def evaluate(model, loader, device):
    model.eval()
    loss_sum, labels, predictions = 0., [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
            labels.extend(y.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return loss_sum / len(labels), accuracy_score(labels, predictions), labels, predictions

def optimizer_for(model, cfg, frozen):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head, body = [], []
    for p in model.parameters():
        is_head = id(p) in head_ids
        p.requires_grad = is_head or not frozen
        (head if is_head else body).append(p)
    groups = [{'params': head, 'lr': cfg.lr}]
    if not frozen:
        groups.append({'params': body, 'lr': cfg.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

def train(cfg):
    if not 0 <= cfg.freeze_epochs < cfg.epochs:
        raise ValueError('Требуется 0 <= freeze_epochs < epochs.')
    seed_everything(cfg.seed)
    runs = Path(cfg.runs)
    runs.mkdir(parents=True, exist_ok=True)
    run = runs / cfg.run_name
    if run.exists():
        raise ValueError(f'Папка {run} уже существует; выберите новое имя запуска.')
    split = split_data(cfg.data, cfg.seed, runs / 'split.json')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device
    model = timm.create_model(cfg.model, pretrained=True, num_classes=len(split['classes']))
    dc = resolve_model_data_config(model)
    spec = {'size': cfg.size, 'crop_pct': dc.get('crop_pct', .875), 'mean': list(dc['mean']), 'std': list(dc['std'])}
    model.to(device)
    run.mkdir()
    write_json(run / 'config.json', asdict(cfg))
    write_json(run / 'environment.json', {'torch': torch.__version__, 'timm': timm.__version__, 'device': str(device)})
    gen = torch.Generator().manual_seed(cfg.seed)
    loaders = {s: DataLoader(Photos(cfg.data, split[s], spec, s == 'train'), batch_size=cfg.batch_size,
                            shuffle=s == 'train', num_workers=0, generator=gen if s == 'train' else None)
               for s in ['train', 'val']}
    frozen = cfg.freeze_epochs > 0
    opt = optimizer_for(model, cfg, frozen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.freeze_epochs if frozen else cfg.epochs))
    best, history = -1., []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        if frozen and epoch == cfg.freeze_epochs:
            frozen = False
            opt = optimizer_for(model, cfg, False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs - epoch)
        model.train()
        if frozen:
            # Зафиксировать статистики BatchNorm замороженного backbone.
            model.eval()
            model.get_classifier().train()
        total, correct, count = 0., 0, 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        vl, va, _, _ = evaluate(model, loaders['val'], device)
        row = {'epoch': epoch + 1, 'train_loss': total/count, 'train_accuracy': correct/count,
               'val_loss': vl, 'val_accuracy': va, 'head_lr': opt.param_groups[0]['lr']}
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(run / 'history.json', history)
        if va > best:
            best = va
            torch.save({'state_dict': model.state_dict(), 'config': asdict(cfg), 'classes': split['classes'],
                        'preprocessing': spec, 'epoch': epoch+1, 'val_accuracy': best}, run / 'best.pt')
        scheduler.step()
    write_json(run / 'summary.json', {'best_val_accuracy': best, 'seconds': time.perf_counter()-start,
                                     'parameters': sum(p.numel() for p in model.parameters())})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for subset in ['train', 'val']:
            ax.plot([h['epoch'] for h in history], [h[f'{subset}_{metric}'] for h in history], label=subset)
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(run / 'learning_curves.png', dpi=160)
    plt.close(fig)
    return run

def finalize(runs):
    """Выбор только по validation; test используется после выбора."""
    runs = Path(runs)
    candidates = sorted(p for p in runs.iterdir() if p.is_dir() and (p / 'summary.json').exists())
    if not candidates:
        raise ValueError('Нет законченных запусков.')
    families = {read_json(p/'config.json')['model'].split('.')[0] for p in candidates}
    if len(families) < 2:
        raise ValueError('Нужно обучить две архитектуры.')
    winner = max(candidates, key=lambda p: read_json(p/'summary.json')['best_val_accuracy'])
    representatives = [max([p for p in candidates if read_json(p/'config.json')['model'].split('.')[0] == family],
                           key=lambda p: read_json(p/'summary.json')['best_val_accuracy']) for family in sorted(families)]
    split = read_json(runs / 'split.json')
    for run in representatives:
        ckpt = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        cfg = TrainConfig(**ckpt['config'])
        model = timm.create_model(cfg.model, pretrained=False, num_classes=len(ckpt['classes']))
        model.load_state_dict(ckpt['state_dict'])
        loader = DataLoader(Photos(cfg.data, split['test'], ckpt['preprocessing']), batch_size=cfg.batch_size)
        loss, acc, y, pred = evaluate(model, loader, 'cpu')
        cm = confusion_matrix(y, pred, labels=list(range(len(ckpt['classes']))))
        write_json(run / 'test_metrics.json', {'accuracy': acc, 'loss': loss, 'confusion_matrix': cm.tolist(),
                   'report': classification_report(y, pred, labels=list(range(len(ckpt['classes']))), target_names=ckpt['classes'], output_dict=True, zero_division=0)})
        write_json(run / 'test_predictions.json', [dict(row, prediction=p) for row, p in zip(split['test'], pred)])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(len(ckpt['classes'])), yticks=range(len(ckpt['classes'])),
               xticklabels=ckpt['classes'], yticklabels=ckpt['classes'], xlabel='Predicted', ylabel='True')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center')
        fig.tight_layout()
        fig.savefig(run / 'confusion_matrix.png', dpi=160)
        plt.close(fig)
    ckpt = torch.load(winner / 'best.pt', map_location='cpu', weights_only=False)
    cfg = TrainConfig(**ckpt['config'])
    model = timm.create_model(cfg.model, pretrained=False, num_classes=len(ckpt['classes']))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    out = BASE / 'models'
    out.mkdir(exist_ok=True)
    write_json(out / 'best_config.json', asdict(cfg))
    write_json(out / 'metadata.json', {'classes': ckpt['classes'], 'preprocessing': ckpt['preprocessing'],
                                      'selected_run': winner.name, 'selected_by': 'validation_accuracy'})
    dummy = torch.zeros(1, 3, cfg.size, cfg.size)
    torch.onnx.export(model, dummy, str(out/'best.onnx'), input_names=['image'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(out/'best.onnx')))
    session = ort.InferenceSession(str(out/'best.onnx'), providers=['CPUExecutionProvider'])
    differences = []
    for row in split['test']:
        with Image.open(Path(cfg.data)/row['path']) as im:
            arr = preprocess(im, ckpt['preprocessing'])[None]
        with torch.no_grad():
            expected = model(torch.from_numpy(arr)).numpy()
        actual = session.run(None, {'image': arr})[0]
        np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-4)
        if expected.argmax(1).item() != actual.argmax(1).item():
            raise RuntimeError('ONNX и PyTorch дают разные классы.')
        differences.append(float(np.max(np.abs(expected-actual))))
    write_json(out/'export_check.json', {'images_checked': len(differences), 'max_abs_error': max(differences)})
    print(f'Выбрана {winner.name}; ONNX сохранён в {out}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        parser.add_argument('--' + f.name.replace('_', '-'), type=f.type, default=getattr(defaults, f.name))
    args = vars(parser.parse_args())
    final = args.pop('finalize')
    cfg = TrainConfig(**args)
    if final:
        finalize(cfg.runs)
    else:
        train(cfg)

if __name__ == '__main__':
    main()
