import numpy as np

from extra.utils import get_child
from models.resnet import ResNet
from models.retinanet import ResNetFPN
from torch.hub import load_state_dict_from_url

from tinygrad import nn
from tinygrad.tensor import Tensor


class MaskRCNN:
  def __init__(self, backbone: ResNet):
    assert isinstance(backbone, ResNet)
    self.backbone = ResNetFPN(backbone, out_channels=256, returned_layers=[1, 2, 3, 4])
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels, 91)

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    return x

  def load_from_pretrained(self):
    self.url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

class RPN:
  def __init__(self, in_channels):
    self.anchor_generator = AnchorGenerator()
    self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location()[0])

  def __call__(self, x):
    pass

class AnchorGenerator:
  def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), strides=(4, 8, 16, 32, 64)):
    anchors = [generate_anchors(stride, (size,), aspect_ratios) for stride, size in zip(strides, sizes)]
    self.cell_anchors = [Tensor(a) for a in anchors]

  def __call__(self, image_list, feature_maps):
    pass

  def num_anchors_per_location(self):
    return [cell_anchors.shape[0] for cell_anchors in self.cell_anchors]

  # anchor generation code below is from the reference implementation here: https://github.com/mlcommons/training/blob/master/object_detection/pytorch/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
def generate_anchors(
  stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
  return _generate_anchors(stride, np.array(sizes, dtype=np.float32) / stride,
                             np.array(aspect_ratios, dtype=np.float32))

  def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
  ):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
      stride,
      np.array(sizes, dtype=np.float) / stride,
      np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
  """Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
  """
  anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
  anchors = _ratio_enum(anchor, aspect_ratios)
  anchors = np.vstack(
    [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
  )
  return anchors


def _whctrs(anchor):
  """Return width, height, x center, and y center for an anchor (window)."""
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack(
    (
      x_ctr - 0.5 * (ws - 1),
      y_ctr - 0.5 * (hs - 1),
      x_ctr + 0.5 * (ws - 1),
      y_ctr + 0.5 * (hs - 1),
    )
  )
  return anchors


def _ratio_enum(anchor, ratios):
  """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """Enumerate a set of anchors for each scale wrt an anchor."""
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


class RPNHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    self.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
    self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
  def __call__(self, x):
    logits = []
    bbox_reg = []
    for feature in x:
      t = self.conv(feature).relu()
      logits.append(self.cls_logits(t))
      bbox_reg.append(self.bbox_pred(t))


class Predictor:
  pass


class PostProcessor:
  pass


class RoIBoxHead:
  def __init__(self, in_channels):
    self.feature_extractor = RoIBoxFeatureExtractor(in_channels)
    self.predictor = Predictor(1024, 2)
    self.post_processor = PostProcessor()

class RoIHeads:
  def __init__(self, in_channels, num_classes):
    self.box = RoIBoxHead(in_channels)
  def __call__(self, features, proposals):
    box_features = self.box(features, proposals)
    return box_features
  def __call__(self, features, proposals):
    x = self.feature_extractor(features, proposals)
    class_logits, box_regression = self.predictor(x)
    return self.post_processor(class_logits, box_regression, proposals)
class Pooler:
  def __init__(self, output_size, scales):
    self.output_size = output_size
    self.scales = scales
class RoIBoxFeatureExtractor:
  def __init__(self, in_channels):
    self.pooler = Pooler(7, 1 / 16)