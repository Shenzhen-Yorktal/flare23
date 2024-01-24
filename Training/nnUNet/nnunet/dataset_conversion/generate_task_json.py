from nnunet.dataset_conversion.utils import generate_dataset_json
import os
from nnunet.paths import *

join = os.path.join


task_name = r'Task123_FLARE23'
target_base = join(nnUNet_raw_data, task_name)
target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")
if not os.path.exists(target_imagesTs):
    os.makedirs(target_imagesTs)
if not os.path.exists(target_labelsTs):
    os.makedirs(target_labelsTs)
maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTr)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTs)

# labels = {
#     }

labels = {0: 'Background', 1: 'Liver', 2: 'Right Kidney', 3: 'Spleen', 4: 'Pancreas',
          5: 'Aorta', 6: 'IVC', 7: 'RAG', 8: 'LAG', 9: 'Gallbladder', 10: 'Esophagus',
          11: 'Stomach', 12: 'Duodenum', 13: 'Left Kidney', 14: 'Tumors'}


description = "FLARE23"

generate_dataset_json(join(target_base, 'dataset.json'),
                      target_imagesTr,
                      target_imagesTs,
                      ('CT',),
                      labels=labels,
                      dataset_name=task_name,
                      dataset_description=description)


