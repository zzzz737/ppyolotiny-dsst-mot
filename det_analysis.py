import paddlex as pdx
from paddlex import transforms as T

pdx.det.draw_pr_curve(
    eval_details_file='output/PPYOLOTiny/best_model/eval_details.json',
    save_dir='output/analysis')

eval_transforms = T.Compose([
    T.Resize(target_size=320, interp='AREA'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='DIC-C2DH-HeLa',
    file_list='DIC-C2DH-HeLa/val_list.txt',
    label_list='DIC-C2DH-HeLa/labels.txt',
    transforms=eval_transforms)

model = pdx.load_model('output/PPYOLOTiny/best_model')
_, evaluate_details = model.evaluate(eval_dataset, return_details=True)
pdx.det.coco_error_analysis(
    gt=evaluate_details['gt'],
    pred_bbox=evaluate_details['bbox'],
    save_dir='output/analysis')
