from .gpu_renderer import ECGRenderer, RenderConfig, RenderBatch
from .gpu_module import ECGImageLayer
from .gpu_parameters import Uniform, Choice, Bernoulli
from .gpu_document import DocumentAssets
from .gpu_layouts import LayoutSpec
from .models_v05 import build_model, task_loss
from .gpu_profiling import profile_layer
from .gpu_supervision import ECGSupervisionLayer, SupervisionConfig, mask_bounding_boxes
from .scenarios import CAPTURE_SCENARIOS, capture_scenario
from .training import detection_targets, padded_detection_targets, masked_segmentation_loss
from .dataset_export import export_yolo_dataset, samples_from_batch
from .assets import load_rgb_bank
from .seeding import render_seed
from .document_templates import DocumentTemplate, DOCUMENT_TEMPLATES
from .thermal_print import ThermalPrint, ThermalPrintConfig
from .physical_camera import PhysicalCamera, LensConfig, SensorConfig
from .emissive_display import EmissiveDisplay, DisplayConfig
from .measured_paper import MeasuredPaperBank, PaperScan
from .material_history import MaterialHistory
from .assets import SceneBank
from .grid_points import GridPointConfig, grid_point_targets, rasterize_grid_points
from .grid_point_learning import grid_point_loss, decode_grid_points

__version__ = "0.12.0"
