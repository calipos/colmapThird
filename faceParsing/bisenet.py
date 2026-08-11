
from __future__ import annotations
import os
import cv2
import numpy as np
import hashlib
# from uniface.constants import ParsingWeights
from log import Logger 
from onnx_utils import create_onnx_session

from base import BaseFaceParser

__all__ = ['BiSeNet']

HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB
def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b''):
            file_hash.update(chunk)
    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        Logger.warning(
            f'Expected hash: {expected_hash}, but got: {actual_hash}')
    return actual_hash == expected_hash

def verify_model_weights(
    modelPath,  # 'resnet34.onnx'
    timeout: int = 60,
    max_retries: int = 3,
) -> str:


    # Lookup model info from registry
    expected_hash = '5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e'


    # Re-download if the cached file is missing or fails verification (e.g. corrupted externally).
    if os.path.exists(modelPath) and expected_hash and verify_file_hash(modelPath, expected_hash):
        return modelPath
    else:
        assert False

    


class BiSeNet(BaseFaceParser):
    """BiSeNet: Bilateral Segmentation Network for Face Parsing with ONNX Runtime.

    BiSeNet is a semantic segmentation model that segments a face image into
    different facial components such as skin, eyes, nose, mouth, hair, etc. The model
    uses a BiSeNet architecture with ResNet backbone and outputs a segmentation mask
    where each pixel is assigned a class label.

    The model supports 19 facial component classes including:
    - Background, skin, eyebrows, eyes, nose, mouth, lips, ears, hair, etc.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Reference:
        https://github.com/yakhyo/face-parsing

    Args:
        model_name (ParsingWeights): The enum specifying the parsing model to load.
            Options: RESNET18, RESNET34.
            Defaults to `ParsingWeights.RESNET18`.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Attributes:
        input_size (tuple[int, int]): Model input dimensions (width, height), read from the ONNX model.
        input_mean (np.ndarray): Per-channel mean values for normalization (ImageNet).
        input_std (np.ndarray): Per-channel std values for normalization (ImageNet).
        mask_type (str): Output type identifier - "class_ids" for BiSeNet.

    Example:
        >>> from uniface.parsing import BiSeNet
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> parser = BiSeNet()
        >>>
        >>> # Detect faces and parse each face
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     bbox = face.bbox
        ...     x1, y1, x2, y2 = map(int, bbox[:4])
        ...     face_crop = image[y1:y2, x1:x2]
        ...     mask = parser.parse(face_crop)
        ...     print(f'Mask shape: {mask.shape}, unique classes: {np.unique(mask)}')
        ...     print(f'Output type: {parser.mask_type}')  # "class_ids"
    """

    mask_type = 'class_ids'

    def __init__(
        self,
        modelPath,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing BiSeNet with model={modelPath}')

        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.providers = providers

        self.model_path = verify_model_weights(modelPath)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(
                self.model_path, providers=self.providers)

            # Get input configuration
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            self.input_name = input_cfg.name
            self.input_size = tuple(
                input_shape[2:4][::-1])  # Update from model

            # Get output configuration
            outputs = self.session.get_outputs()
            self.output_names = [output.name for output in outputs]

            Logger.info(
                f'BiSeNet initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(
                f"Failed to load parsing model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(
                f'Failed to initialize parsing model: {e}') from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess a face image for parsing.

        Args:
            face_image (np.ndarray): A face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, self.input_size,
                           interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and apply normalization
        image = image.astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std

        # HWC -> CHW -> NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Postprocess model output to segmentation mask.

        Args:
            outputs (np.ndarray): Raw model output.
            original_size (tuple[int, int]): Original image size (width, height).

        Returns:
            np.ndarray: Segmentation mask resized to original dimensions.
        """
        # Get the class with highest probability for each pixel
        predicted_mask = outputs.squeeze(0).argmax(0).astype(np.uint8)

        # Resize back to original size
        restored_mask = cv2.resize(
            predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        return restored_mask

    def parse(self, image: np.ndarray, *, landmarks: np.ndarray | None = None) -> np.ndarray:
        """Perform end-to-end face parsing on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the segmentation mask.

        BiSeNet operates on face crops and does not require landmarks.
        The `landmarks` parameter is accepted for API compatibility but ignored.

        Args:
            image (np.ndarray): A face image in BGR format.
            landmarks (np.ndarray | None): Ignored. Accepted for interface
                compatibility with `BaseFaceParser`.

        Returns:
            np.ndarray: Segmentation mask with the same size as input image.
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        input_tensor = self.preprocess(image)
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], original_size)


FACE_PARSING_COLORS = [
    [0, 0, 0],
    [0, 85, 255],
    [0, 170, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [85, 255, 0],
    [170, 255, 0],
    [255, 0, 0],
    [255, 0, 85],
    [255, 0, 170],
    [255, 85, 0],
    [255, 170, 0],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
    [255, 0, 255],
]
def vis_parsing_maps(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    save_image: bool = False,
    save_path: str = 'result.png',
) -> np.ndarray:
    """Visualize face parsing segmentation mask by overlaying colored regions.

    Args:
        image: Input face image in BGR format with shape `(H, W, 3)`.
        segmentation_mask: Segmentation mask with shape `(H, W)` where each
            pixel value represents a facial component class (0-18).
        save_image: Whether to save the visualization to disk. Defaults to False.
        save_path: Path to save the visualization if *save_image* is True.

    Returns:
        Blended image with segmentation overlay in BGR format.

    Example:
        >>> import cv2
        >>> from uniface.parsing import BiSeNet
        >>> from uniface.draw import vis_parsing_maps
        >>> parser = BiSeNet()
        >>> face_image = cv2.imread('face.jpg')
        >>> mask = parser.parse(face_image)
        >>> result = vis_parsing_maps(face_image, mask)
        >>> cv2.imwrite('parsed_face.jpg', result)
    """
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Create a color mask in BGR; the palette is padded to max_class + 1 so class ids
    # beyond the color table map to black instead of raising
    max_class = int(segmentation_mask.max())
    palette = np.zeros(
        (max(max_class + 1, len(FACE_PARSING_COLORS)), 3), dtype=np.uint8)
    palette[: len(FACE_PARSING_COLORS)] = FACE_PARSING_COLORS
    segmentation_mask_color = palette[segmentation_mask]

    # Blend image and color mask directly (both in BGR format)
    blended_image = cv2.addWeighted(
        image, 0.6, segmentation_mask_color, 0.4, 0)

    if save_image:
        cv2.imwrite(save_path, blended_image, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image
