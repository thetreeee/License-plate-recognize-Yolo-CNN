"""CPU smoke test for the standalone layout notebook; no downloads or Drive writes."""
import ast
import io
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/MobileNetV3_Small_VNLP_Layout_Finetune.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    imports, definitions = [], []
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = '\n'.join(line for line in ''.join(cell['source']).splitlines() if not line.startswith('%'))
        tree = ast.parse(source)
        compile(tree, str(notebook_path), 'exec')
        count += 1
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module == 'google.colab':
                    continue
                imports.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions.append(node)

    env = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(notebook_path), 'exec'), env)
    torch, np, pd = env['lyt_torch'], env['lyt_np'], env['lyt_pd']
    torch.set_num_threads(2)
    env.update(lyt_size=128, lyt_seed=42, lyt_batch_size=4, lyt_workers=0,
               lyt_device=torch.device('cpu'), lyt_bbox_mode='xyxy', lyt_crop_margin=.03,
               lyt_class_names=['one_row', 'two_rows'], lyt_label_map={'one_row':0, 'two_rows':1})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook_path), 'exec'), env)
    assert env['lyt_parse_valid'](pd.Series(['True','FALSE',1,0])).tolist() == [True,False,True,False]
    try:
        env['lyt_parse_valid'](pd.Series(['unknown']))
    except AssertionError:
        pass
    else:
        raise AssertionError('Malformed valid flag accepted')

    rgb = np.full((80,240,3), 220, dtype=np.uint8)
    row = dict(image_id='sample.jpg', a=10,b=10,c=230,d=70)
    crop = env['lyt_crop_plate'](rgb,row)
    assert crop.shape == (64,234,3), crop.shape
    letterboxed = env['lyt_letterbox'](env['lyt_Image'].fromarray(crop))
    assert letterboxed.size == (128,128)
    env['lyt_bbox_mode']='xywh'
    assert np.array_equal(crop,env['lyt_crop_plate'](rgb,dict(row,c=220,d=60)))
    env['lyt_bbox_mode']='xyxy'
    try:
        env['lyt_crop_plate'](rgb,dict(row,c=999))
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-bounds bbox silently accepted')

    # Execute only the two transform assignments from the dataset cell.
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree=ast.parse(''.join(cell['source'])) if not ''.join(cell['source']).startswith('%') else None
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(
                isinstance(t,ast.Name) and t.id in {'lyt_train_augment','lyt_to_tensor'}
                for t in node.targets
            ):
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(notebook_path),'exec'),env)
    env['lyt_read_rgb']=lambda _:rgb.copy()
    rows=[]
    for i in range(4):
        rows.append(dict(row,image_id=f'{i}.jpg',path='mock.jpg',valid=i!=3,
                         layout='one_row' if i%2==0 else 'two_rows'))
    frame=pd.DataFrame(rows)
    loader=env['lyt_make_loader'](frame)
    model=env['lyt_mobilenet'](weights=None)
    model.classifier[-1]=env['lyt_nn'].Linear(model.classifier[-1].in_features,2)
    env['lyt_criterion']=env['lyt_nn'].CrossEntropyLoss()
    metrics,details,_=env['lyt_run_epoch'](model,loader)
    assert metrics['samples']==4 and metrics['invalid_inputs']==1
    assert details.loc[~details.valid,'prediction'].tolist()==[-1]
    assert np.isfinite(metrics['loss_valid_inputs'])
    check,_=env['lyt_metrics'](pd.DataFrame(dict(target=[0,1],prediction=[0,-1],valid=[True,False])))
    assert check['accuracy']==.5 and check['macro_f1']==.5

    train_loader=env['lyt_make_loader'](frame[frame.valid],True)
    model.features.requires_grad_(False)
    bn=next(m for m in model.features.modules() if isinstance(m,env['lyt_nn'].BatchNorm2d))
    bn_before=bn.running_mean.clone()
    head_before=model.classifier[-1].weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    env['lyt_run_epoch'](model,train_loader,optimizer,freeze_backbone=True)
    assert torch.equal(bn_before,bn.running_mean), 'Frozen backbone BatchNorm changed'
    assert not torch.equal(head_before,model.classifier[-1].weight)
    model.features.requires_grad_(True)
    env['lyt_run_epoch'](model,train_loader,optimizer)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.features.parameters())

    stream=io.BytesIO()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=1),stream)
    stream.seek(0)
    state=torch.load(stream,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])

    # Verify the embedded OCR model/preprocessing exactly match the CRNN notebook.
    source_path=notebook_path.with_name('CRNN_BiLSTM_CTC_VNLP_From_Scratch.ipynb')
    source_nb=json.loads(source_path.read_text(encoding='utf-8'))
    baseline_definitions=[]
    for cell in source_nb['cells']:
        if cell['cell_type']!='code':
            continue
        source='\n'.join(line for line in ''.join(cell['source']).splitlines() if not line.startswith('%'))
        for node in ast.parse(source).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in {
                'CRNN','row_separator','merge_rows','letterbox'
            }:
                baseline_definitions.append(node)
    baseline=dict(nn=env['lyt_nn'],NUM_CLASSES=37,np=np,cv2=env['joint_cv2'],HEIGHT=32,WIDTH=256)
    exec(compile(ast.Module(body=baseline_definitions,type_ignores=[]),str(source_path),'exec'),baseline)
    reference_crnn=baseline['CRNN']().eval()
    embedded_crnn=env['JointCRNN'](37).eval()
    embedded_crnn.load_state_dict(reference_crnn.state_dict(),strict=True)
    x=torch.randn(2,3,32,256)
    with torch.inference_mode():
        assert torch.equal(reference_crnn(x),embedded_crnn(x)), 'Embedded architecture changed'

    env.update(joint_height=32,joint_width=256,joint_idx_to_char={
        i+1:c for i,c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    })
    cv2=env['joint_cv2']
    two_rows=np.full((120,240,3),255,np.uint8)
    cv2.putText(two_rows,'29A1',(20,43),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2)
    cv2.putText(two_rows,'12345',(12,104),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2)
    old_cut=baseline['row_separator'](two_rows)
    new_cut=env['joint_row_separator'](two_rows)
    assert np.array_equal(old_cut[0],new_cut[0]) and old_cut[1:3]==new_cut[1:3]
    old_image=baseline['letterbox'](baseline['merge_rows'](two_rows,old_cut[0]))[0]
    new_image=env['joint_letterbox'](env['joint_merge_rows'](two_rows,new_cut[0]))[0]
    assert np.array_equal(old_image,new_image), 'Embedded row preprocessing changed'
    tensor,ok,_=env['joint_ocr_input'](two_rows,'two_rows')
    assert tensor.shape==(3,32,256) and ok
    assert torch.isfinite(tensor).all() and tensor.min()>=-1 and tensor.max()<=1

    model.eval()
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([10.,-10.]))
    env.update(joint_loaded=True,joint_crnn=embedded_crnn,joint_layout_model=model)
    paired_frame=frame.copy()
    paired_frame['layout']='one_row'
    paired_frame['text']='36A12345'
    details=env['joint_evaluate'](paired_frame,batch_size=2)
    assert len(details)==4 and int((~details.crop_valid).sum())==1
    assert (details.reference_prediction==details.predicted_layout_prediction).all()
    assert (details.loc[~details.crop_valid,'reference_prediction']=='').all()
    report=env['joint_comparison'](details)
    assert report['accuracy_drop_percentage_points']==0
    assert report['cer_increase_percentage_points']==0
    assert sum(report['transitions'].values())==4
    result=env['joint_predict_crop'](crop)
    assert result['predicted_layout']=='one_row' and isinstance(result['text'],str)
    components=sum(result[k] for k in ['layout_ms','preprocess_ms','crnn_decode_ms'])
    assert abs(result['total_ms']-components)<1e-6 and result['total_ms']>0
    speed=env['joint_speed_summary'](pd.DataFrame([result]),'test')
    assert speed['total_p95_ms']>0 and speed['crops_per_second']>0
    print(f'PASS: {count} code cells compile; classifier training/checkpoint, exact CRNN/preprocessing parity, paired OCR denominator/deltas, timed pipeline.')


if __name__=='__main__':
    main()
