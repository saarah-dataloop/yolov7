from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from string import Template
from pathlib import Path
import dtlpy as dl
import numpy as np
import traceback
import logging
import shutil
import torch
import tqdm
import json
import yaml
import os

from utils.callbacks import Callbacks
from utils.augmentations import letterbox
from utils.general import increment_path, non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync

logger = logging.getLogger('yolo-v5')
logging.basicConfig(level='INFO')


class ModelAdapter(dl.BaseModelAdapter):
    """
    Yolo5 Model adapter - based on ultralytics pytorch implementation.
    The class bind Dataloop model and snapshot entities with model code implementation


    # NOTE: Starting dtlpy version 1.35 we use a different BaseModelAdapter
            This is the updated version of the adapter for dtlpy 1.35
    """

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: `str` directory path in local fileSystem where the weights is taken from
        """
        t1 = time_sync()
        weights = self.configuration.get('weights_filename', 'yolov5s.pt')  # model.pt path(s)
        half = self.configuration.get('half', False)  # use FP16 half-precision inference
        device = '0' if torch.cuda.is_available() else 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        weights_filepath = os.path.join(local_path, weights)

        # Load model
        device = select_device(device)
        logger.info('device is: {}'.format(device))
        model = DetectMultiBackend(weights_filepath, device=device)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        # Half
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        self.model = model
        self.device = device
        self.half = half
        logger.info('Model Load Speed: {:.1f}ms'.format(1e3 * (time_sync() - t1)))

    def preprocess(self, x):
        # Padded resize
        img = letterbox(x,
                        self.configuration['img_size'],
                        stride=self.model.stride,
                        auto=self.model.pt
                        )[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(float)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of image
        :param batch: `np.ndarray` NCHW
        :return `list[dl.AnnotationCollection]` prediction results by len(batch)
        """
        img_size = self.configuration['img_size']
        # stride = self.configuration['stride']
        # auto = self.configuration['auto']
        conf_thres = self.configuration['conf_thres']
        iou_thres = self.configuration['iou_thres']
        agnostic_nms = self.configuration['agnostic_nms']
        max_det = self.configuration['max_det']
        id_to_label_map = self.snapshot.id_to_label_map

        seen = batch.shape[0]
        dt = [0.0, 0.0, 0.0]
        # Run inference
        # self.model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        t1 = time_sync()

        preprocessed_batch = torch.from_numpy(np.asarray([self.preprocess(img) for img in batch])).to(self.device)
        preprocessed_batch = preprocessed_batch.half() if self.half else preprocessed_batch.float()  # uint8 to fp16/32
        logger.info('[preprocess]: model batch size{}'.format(preprocessed_batch.shape))

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(preprocessed_batch)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        batch_annotations = list()
        for i_img, det in enumerate(pred):  # per image
            img_shape = batch[i_img].shape
            p_img_shape = preprocessed_batch[i_img].shape[1:]
            det[:, :4] = scale_coords(p_img_shape, det[:, :4], img_shape).round()
            image_annotations = dl.AnnotationCollection()
            for *xyxy, conf, cls in reversed(det):
                image_annotations.add(annotation_definition=dl.Box(left=xyxy[0],
                                                                   top=xyxy[1],
                                                                   right=xyxy[2],
                                                                   bottom=xyxy[3],
                                                                   label=id_to_label_map[int(cls)]
                                                                   # when loading snapshot, json treats keys as str
                                                                   ),
                                      model_info={'name': self.model_name,
                                                  'confidence': conf})
            batch_annotations.append(image_annotations)

        # Process predictions

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        logger.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *img_size)}' % t)
        return batch_annotations

    def save(self, local_path, **kwargs):
        """
         saves configuration and weights locally

              Virtual method - need to implement

              the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

          :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', self.configuration['weights_filename'])
        weights_path = os.path.join(local_path, weights_filename)
        torch.save(self.model, weights_path)
        self.configuration['weights_filename'] = weights_filename

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        import train as train_script
        if os.path.isfile(self.configuration['hyp_yaml_fname']):
            hyp_full_path = self.configuration['hyp_yaml_fname']
        else:
            hyp_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'hyps',
                                         self.configuration['hyp_yaml_fname'])
        hyp = yaml.safe_load(open(hyp_full_path, 'r', encoding='utf-8'))
        opt = self._create_opt(data_path=data_path, output_path=output_path, **kwargs)
        logger.info("Created OPT configuration: batch_size {b};  num_epochs {num} image_size {sz}".
                    format(b=opt.batch_size, num=opt.epochs, sz=opt.imgsz))
        logger.debug("OPT config full debug: {}".format(opt))
        # Make sure opt.weights has the exact model file as it will load from there
        train_results = train_script.train(hyp, opt, self.device, callbacks=Callbacks())
        logger.info('Train Finished. Actual output path: {}'.format(opt.save_dir))

        # load best model weights
        best_model_wts = os.path.join(opt.save_dir, 'weights', 'best.pt')
        self.model = torch.load(best_model_wts, map_location=self.device)['model']
        # self.model.load_state_dict(best_model_wts)

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """

        # update the label_map {id: label} to the one from the snapshot
        id_to_label_map = self.snapshot.id_to_label_map
        label_to_id_map = {v: k for k, v in id_to_label_map.items()}
        # White / Black list option to use
        white_list = kwargs.get('white_list', False)  # white list is the verified annotations labels to work with
        black_list = kwargs.get('black_list', False)  # black list is the illegal annotations labels to work with
        empty_prob = kwargs.get('empty_prob', 0)  # do we constraint number of empty images

        for partiton in dl.SnapshotPartitionType:
            in_labels_path = os.path.join(data_path, partiton, 'json')
            in_images_path = os.path.join(data_path, partiton, 'items')

            # Train - Val split
            labels_path = os.path.join(data_path, partiton, 'labels')
            images_path = os.path.join(data_path, partiton, 'images')

            # TODO: currently the function is called inside partition loop - need to fix
            if os.path.isdir(labels_path):
                if len(os.listdir(labels_path)) > 0:
                    logger.warning('dir {} already been processed. Skipping'.format(labels_path))
                    continue
            else:
                os.makedirs(labels_path, exist_ok=True)
                os.makedirs(images_path, exist_ok=True)

            # set the list of files to parse and convert
            json_filepaths = list()
            for path, subdirs, files in os.walk(in_labels_path):
                # break
                for fname in files:
                    filename, ext = os.path.splitext(fname)
                    if ext.lower() not in ['.json']:
                        continue
                    json_filepaths.append(os.path.join(path, fname))
            np.random.shuffle(json_filepaths)

            counters = {
                'empty_items_found': 0,
                'empty_items_discarded': 0,
                'corrupted_cnt': 0
            }
            pool = ThreadPool(processes=16)
            lock = Lock()
            for in_json_filepath in tqdm.tqdm(json_filepaths, unit='file'):
                pool.apply_async(func=self._parse_single_annotation_file,
                                 args=(in_json_filepath, in_labels_path, labels_path,
                                       in_images_path, images_path, label_to_id_map, counters, lock),
                                 kwds={'white_list': white_list,
                                       'black_list': black_list,
                                       'empty_prob': empty_prob}
                                 )
            pool.close()
            pool.join()
            pool.terminate()

        config_path = os.path.join(data_path, self.configuration['data_yaml_fname'])
        self.create_yaml(
            train_path=os.path.join(data_path, dl.SnapshotPartitionType.TRAIN),
            val_path=os.path.join(data_path, dl.SnapshotPartitionType.VALIDATION),
            classes=list(label_to_id_map.keys()),
            config_path=config_path,
        )

        train_cnt = sum([len(files) for r, d, files in os.walk(data_path + '/train/labels')])
        val_cnt = sum([len(files) for r, d, files in os.walk(data_path + '/validation/labels')])

        msg = "Finished converting the data. Creating config file: {!r}. ".format(config_path) + \
              "\nLabels dict {}.\nlabel_map   {}".format(label_to_id_map, id_to_label_map) + \
              "\nVal count   : {}\nTrain count: {}".format(val_cnt, train_cnt)
        logger.info(msg)

    def _parse_single_annotation_file(self, in_json_filepath, in_labels_path, labels_path,
                                      in_images_path, images_path, label_to_id, counters, lock,
                                      white_list=False, black_list=False, empty_prob=0):
        try:
            # read the item json
            with open(in_json_filepath, 'r') as f:
                data = json.load(f)
            annotations = dl.AnnotationCollection.from_json(_json=data['annotations'])
            if 'itemMetadata' in data:  # support both types of json files
                item_metadata = data['itemMetadata']
            else:
                item_metadata = data['metadata']

            # partition = item_metadata['system']['snapshotPartition']
            img_width, img_height = item_metadata['system']['width'], item_metadata['system']['height']

            output_txt_filepath = in_json_filepath.replace(in_labels_path, labels_path).replace('.json', '.txt')
            os.makedirs(os.path.dirname(output_txt_filepath), exist_ok=True)
            item_lines = list()
            for ann in annotations:
                if ann.type == 'box':

                    # skip annotation if on white / black list
                    if white_list and ann.label not in white_list:
                        continue
                    if black_list and ann.label in black_list:
                        continue

                    a_h = round(ann.bottom - ann.top, 5)
                    a_w = round(ann.right - ann.left, 5)
                    x_c = round(ann.left + (a_w / 2), 5)
                    y_c = round(ann.top + (a_h / 2), 5)
                    label = ann.label
                    with lock:
                        if label not in label_to_id:
                            label_to_id[label] = len(label_to_id)
                    label_id = label_to_id[label]
                    line = '{label_id} {x_center} {y_center} {width} {height}'.format(
                        label_id=label_id, x_center=x_c / img_width, y_center=y_c / img_height,
                        width=a_w / img_width, height=a_h / img_height)
                    item_lines.append(line)

            if len(item_lines) == 0:
                with lock:
                    counters['empty_items_found'] += 1
                if empty_prob > 0 and np.random.random() < empty_prob:  # save empty image with some prob
                    with lock:
                        counters['empty_items_discarded'] += 1
                    return

            # Create new files in the train-set
            dst = images_path + data['filename']
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src=in_images_path + data['filename'], dst=dst)
            with open(output_txt_filepath, 'w') as f:
                f.write('\n'.join(item_lines))
                f.write('\n')
        except Exception:
            with lock:
                counters['corrupted_cnt'] += 1
            logger.error("file: {} had problem. Skipping\n\n{}".format(in_json_filepath, traceback.format_exc()))

    def create_yaml(self, train_path, val_path, classes, config_path='/tmp/dlp_data.yaml'):
        """
        Create the data (or is it the config) yaml
        """

        yaml_template = Path(Path(__file__).parent.parent.absolute(), 'data_yaml_template.txt')
        template = Template(yaml_template.open('r').read())
        yaml_str = str({
            'train': train_path,
            'val': val_path,
            'nc': len(classes),
            'names': classes
        })

        with open(config_path, 'w') as f:
            f.write(yaml_str)

    def _create_opt(self, data_path, output_path, **kwargs):
        import argparse
        data_yaml_path = os.path.join(data_path, self.configuration['data_yaml_fname'])
        if kwargs.get('auto_increase', False) and os.path.isdir(output_path):
            output_path = increment_path(Path(output_path)).as_posix()

        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', type=str, default=output_path, help='path to save the results')
        parser.add_argument('--epochs', type=int, default=self.configuration.get('num_epochs', 100))  # 300
        parser.add_argument('--batch-size', type=int, default=self.configuration.get('batch_size', 4),
                            help='batch size for all GPUs')
        # parser.add_argument('--total-batch-size',  type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--weights', type=str, default=self.configuration['weights_filename'],
                            help='initial weights file name')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=max(self.configuration['img_size']),
                            help='train, val image size (pixels)')
        parser.add_argument('--data', type=str, default=data_yaml_path, help='dlp_data.yaml path')
        parser.add_argument('--workers', type=int, default=self.configuration.get('workers', 0),
                            help='maximum number of dataloader workers')

        parser.add_argument('--global_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')

        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--cache', type=str, nargs='?', const='ram',
                            help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
        parser.add_argument('--patience', type=int, default=100,
                            help='EarlyStopping patience (epochs without improvement)')

        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                            help='Freeze layers: backbone=10, first3=0 1 2')

        # NEW
        # parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
        # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # parser.add_argument('--project', default='runs/train', help='save to project/name')
        # parser.add_argument('--name', default='exp', help='save to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        # parser.add_argument('--entity', default=None, help='W&B entity')
        # parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
        # parser.add_argument('--bbox_interval', type=int, default=-1,
        #                     help='Set bounding-box image logging interval for W&B')
        # parser.add_argument('--artifact_alias', type=str, default="latest",
        #                     help='version of dataset artifact to be used')

        opt = parser.parse_known_args()
        return opt[0]


def model_creation(project_name):
    project = dl.projects.get(project_name)
    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/yolov5.git',
                              git_tag='dtlpy-v6.1.1')
    model = project.models.create(model_name='yolo-v5',
                                  description='Global Dataloop Yolo V5 implemented in pytorch',
                                  output_type=dl.AnnotationType.BOX,
                                  tags=['torch', 'yolo', 'detection'],
                                  codebase=codebase,
                                  entry_point='model_adapter.py',
                                  default_runtime=dl.KubernetesRuntime(
                                      autoscaler=dl.KubernetesRabbitmqAutoscaler(),
                                      runner_image='dataloop_runner-cpu/yolov5-openvino:1'),
                                  default_configuration={'weights_filename': 'yolov5s.pt',
                                                         'img_size': [640, 640],
                                                         'conf_thres': 0.25,
                                                         'iou_thres': 0.45,
                                                         'max_det': 1000,
                                                         'device': 'cuda',
                                                         'agnostic_nms': False,
                                                         'half': False},
                                  )
    return model


def snapshot_creation(model: dl.Model, yolo_size='small'):
    # TODO: can we add two model arc in one dir - yolov5l, yolov5s
    # Select the specific arch and gcs bucket
    if yolo_size == 'small':
        gcs_prefix = 'yolo-v5/small'
        weights_filename = 'yolov5s.pt'
    elif yolo_size == 'large':
        gcs_prefix = 'yolo-v5-v6/large'
        weights_filename = 'yolov5l6.pt'
    elif yolo_size == 'extra':
        gcs_prefix = 'yolo-v5-v6/extra'
        weights_filename = 'yolov5x6.pt'
    elif yolo_size == 'openvino':
        gcs_prefix = 'yolo-v5-v6/openvino'
        weights_filename = 'yolov5s6.xml'

    else:
        raise RuntimeError('yolo_size {!r} - un-supported, choose "small" "large" or "extra" '.format(yolo_size))
    with open('data/coco.yaml') as f:
        coco_yaml = yaml.safe_load(f)
    labels = coco_yaml['names']
    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix=gcs_prefix)
    snapshot = model.snapshots.create(snapshot_name='pretrained-yolo-v5-{}'.format(yolo_size),
                                      description='yolo v5 {} arch, pretrained on ms-coco'.format(yolo_size),
                                      tags=['pretrained', 'ms-coco'],
                                      dataset_id=None,
                                      status='trained',
                                      configuration={
                                          'weights_filename': weights_filename,
                                          'img_size': [640, 640],
                                          'conf_thres': 0.25,
                                          'iou_thres': 0.45,
                                          'max_det': 1000,
                                          'device': 'cuda',
                                          'agnostic_nms': False,
                                          'half': False,
                                          'data_yaml_fname': 'coco.yaml',
                                          'hyp_yaml_fname': 'hyp.finetune.yaml',
                                          'id_to_label_map': {ind: label for ind, label in enumerate(labels)}},
                                      project_id=model.project.id,
                                      bucket=bucket,
                                      labels=labels
                                      )
    return snapshot
