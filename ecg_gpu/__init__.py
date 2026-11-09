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
from .training import lead_name_detection_targets
from .dataset_export import export_yolo_dataset, samples_from_batch
from .paired_dataset import paired_samples_from_render, export_paired_dataset
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
from .convenience import prepare_signals, render_ecg, to_pil, describe_parameters
from .schema_v05 import LEADS
from .report_text import decode_report_text

__version__ = "0.32.0"

__all__ = [
    'ECGRenderer',
    'RenderConfig',
    'RenderBatch',
    'ECGImageLayer',
    'Uniform',
    'Choice',
    'Bernoulli',
    'DocumentAssets',
    'LayoutSpec',
    'build_model',
    'task_loss',
    'profile_layer',
    'ECGSupervisionLayer',
    'SupervisionConfig',
    'mask_bounding_boxes',
    'CAPTURE_SCENARIOS',
    'capture_scenario',
    'detection_targets',
    'padded_detection_targets',
    'masked_segmentation_loss',
    'export_yolo_dataset',
    'samples_from_batch',
    'load_rgb_bank',
    'render_seed',
    'DocumentTemplate',
    'DOCUMENT_TEMPLATES',
    'ThermalPrint',
    'ThermalPrintConfig',
    'PhysicalCamera',
    'LensConfig',
    'SensorConfig',
    'EmissiveDisplay',
    'DisplayConfig',
    'MeasuredPaperBank',
    'PaperScan',
    'MaterialHistory',
    'SceneBank',
    'GridPointConfig',
    'grid_point_targets',
    'rasterize_grid_points',
    'grid_point_loss',
    'decode_grid_points',
    'prepare_signals',
    'render_ecg',
    'to_pil',
    'describe_parameters',
    'LEADS',
    'decode_report_text',
    'lead_name_detection_targets',
    'paired_samples_from_render',
    'export_paired_dataset',
    '__version__',
]
