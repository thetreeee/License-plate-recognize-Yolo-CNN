"""Local CPU smoke test; does not mount Drive, train fully, or change dataset files.

Usage: python scripts/validate_crnn_notebook.py [--zip D:/Downloads/VNLP_detection.zip]
Requires torch, numpy, pandas, pillow, matplotlib, opencv-python, tqdm.
"""
import argparse
import ast
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--qa', type=Path, help='Optional output contact sheet for visual inspection')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'notebooks/CRNN_BiLSTM_CTC_VNLP_From_Scratch.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    definitions = []
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        source = '\n'.join(line for line in source.splitlines() if not line.startswith('%'))
        tree = ast.parse(source)
        compile(tree, str(path), 'exec')
        count += 1
        definitions.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    env = dict(np=np, pd=pd, cv2=cv2, torch=torch, nn=nn, Dataset=Dataset,
               DataLoader=DataLoader, Image=Image, random=random, Path=Path, tqdm=tqdm,
               plt=plt, SEED=42, BBOX_MODE='xyxy', CROP_MARGIN=.03,
               HEIGHT=32, WIDTH=256, NUM_CLASSES=37, BATCH_SIZE=2,
               NUM_WORKERS=0, DEVICE=torch.device('cpu'), USE_AMP=False,
               CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', BLANK=0, overrides={})
    env['char_to_idx'] = {c:i+1 for i,c in enumerate(env['CHARSET'])}
    env['idx_to_char'] = {i+1:c for i,c in enumerate(env['CHARSET'])}
    env['criterion'] = nn.CTCLoss(blank=0, zero_infinity=False)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), env)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = env['CRNN']()
    x = torch.randn(2,3,32,256)
    labels = ['111', '29A112345']
    targets = torch.tensor([env['char_to_idx'][c] for s in labels for c in s])
    logits = model(x)
    assert logits.shape == (64,2,37), logits.shape
    loss = env['ctc_loss'](logits, targets, torch.tensor([3,9]), [{'text':s} for s in labels])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0
    assert env['minimum_ctc_steps']('111') == 5
    seq = [1,1,0,1,2,2,0]
    fake = torch.full((len(seq),1,37), -20.)
    for t,c in enumerate(seq): fake[t,0,c] = 20
    assert env['greedy_decode'](fake) == ['001']
    assert env['edit_distance']('29A1', '29A') == 1
    image = np.full((120,240,3),255,np.uint8)
    cv2.putText(image,'29A1',(20,43),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2)
    cv2.putText(image,'12345',(12,104),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2)
    cut,ok,method,_,_ = env['row_separator'](image)
    assert ok and 44 < np.median(cut) < 80, (cut,ok,method)
    merged = env['merge_rows'](image,cut)
    prepared,_,_ = env['letterbox'](merged)
    assert prepared.shape == (32,256,3)
    blank = np.full((120,240,3),255,np.uint8)
    assert not env['row_separator'](blank)[1], 'Blank image marked confident'
    try:
        env['crop_plate'](image,dict(a=1,b=1,c=999,d=20))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid bbox silently accepted')
    # Check validation metrics keep invalid inputs in their denominators.
    sample_rows = [dict(image_id=str(i),text='111',layout='one_row',valid=(i==0),
                        split_ok=True,quality_review=False) for i in range(2)]
    batch = (x, torch.tensor([2]*6), torch.tensor([3,3]), sample_rows)
    metrics,details = env['run_epoch'](model,[batch])
    assert metrics['samples']==2 and metrics['invalid_inputs']==1
    assert details.iloc[1].prediction==''
    train_rows = [dict(row,valid=True) for row in sample_rows]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    before=model.head.weight.detach().clone()
    train_batch=(x,torch.tensor([2]*6),torch.tensor([3,3]),train_rows)
    env['run_epoch'](model,[train_batch],optimizer,scaler)
    assert not torch.equal(before,model.head.weight), 'Training failed to update weights'
    print(f'PASS: {count} code cells compile; CRNN shape, CTC backward, decoder, synthetic rows, invalid accounting.')
    if args.zip:
        check_zip(args.zip, env, args.qa)


def check_zip(path, env, qa=None):
    cache = {}
    samples = []
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.jpg')]
        for folder in ['one_row','two_rows','two_rows_label_xe_may']:
            subset = [n for n in names if n.split('/')[-2] == folder]
            for name in random.Random(42).sample(subset,min(40,len(subset))):
                with Image.open(io.BytesIO(archive.read(name))) as im:
                    cache[name] = np.asarray(im.convert('RGB')).copy()
                parts = Path(name).stem.split('_')
                a,b,c,d = map(int,parts[-4:])
                samples.append(dict(path=name,image_id=name,text=parts[-5].upper(),
                                    layout='one_row' if folder=='one_row' else 'two_rows',
                                    a=a,b=b,c=c,d=d,folder=folder))
    env['read_rgb'] = lambda path: cache[path].copy()
    results=[]
    for row in samples:
        try:
            image,info = env['preprocess'](row)
            assert image.shape==(32,256,3)
            results.append(dict(folder=row['folder'],valid=True,confident=info['split_ok']))
        except ValueError:
            results.append(dict(folder=row['folder'],valid=False,confident=False))
    report=pd.DataFrame(results).groupby('folder').agg(samples=('valid','size'),valid=('valid','sum'),confident=('confident','sum'))
    print(report.to_string())
    assert sum(r['valid'] for r in results) >= .9*len(results), 'Too many invalid crops: verify bbox interpretation'
    print('PASS: actual ZIP sample processing; confidence is heuristic, not measured split accuracy.')
    if qa:
        chosen = [r for r in samples if r['layout']=='two_rows'][:12]
        fig,axes=plt.subplots(len(chosen),3,figsize=(14,len(chosen)*2.3))
        for axs,row in zip(axes,chosen):
            image,info,debug=env['preprocess'](row,True)
            axs[0].imshow(debug['crop']); axs[0].plot(debug['cut_line'],color='red')
            axs[0].set_title(row['text']+' | '+info['split_method'])
            axs[1].imshow(debug['mask'],cmap='gray')
            axs[2].imshow(image)
            for ax in axs: ax.axis('off')
        fig.tight_layout(); fig.savefig(qa,dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
