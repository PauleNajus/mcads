from unittest.mock import patch, MagicMock
from django.test import SimpleTestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from .image_processing import looks_like_dicom
from .models import XRayImage
from .model_loader import load_model, clear_model_cache

class ImageProcessingTests(SimpleTestCase):
    def test_looks_like_dicom_header(self):
        # Create a header with "DICM" at offset 128
        header = b"\x00" * 128 + b"DICM"
        self.assertTrue(looks_like_dicom("test.dcm", header))
        self.assertTrue(looks_like_dicom(None, header))

    def test_looks_like_dicom_extension(self):
        header = b"\x00" * 132 # Invalid header
        self.assertTrue(looks_like_dicom("test.dcm", header))
        self.assertTrue(looks_like_dicom("TEST.DICOM", header))
        self.assertFalse(looks_like_dicom("test.png", header))

    def test_looks_like_dicom_negative(self):
        header = b"\x00" * 132
        self.assertFalse(looks_like_dicom("test.png", header))
        self.assertFalse(looks_like_dicom(None, header))

class XRayImageModelTests(SimpleTestCase):
    def test_create_xray_image(self):
        # Create a dummy image
        image = SimpleUploadedFile("test_image.jpg", b"file_content", content_type="image/jpeg")
        xray = XRayImage(
            image=image,
            first_name="John",
            last_name="Doe",
            patient_id="12345",
            uploaded_at=timezone.now(),
        )
        self.assertEqual(xray.first_name, "John")
        self.assertIn("John Doe", str(xray))
        self.assertIn("12345", str(xray))
        self.assertEqual(xray.processing_status, 'pending')

    def test_calculate_severity_level(self):
        # Severity is derived from the *maximum* predicted pathology probability
        # (Manchester Triage System approximation):
        # 1=Immediate (>=0.80) ... 5=Non-urgent (<0.20)
        xray = XRayImage(atelectasis=0.19)
        self.assertEqual(xray.calculate_severity_level, 5)

        xray.atelectasis = 0.20
        self.assertEqual(xray.calculate_severity_level, 4)

        xray.atelectasis = 0.40
        self.assertEqual(xray.calculate_severity_level, 3)

        xray.atelectasis = 0.60
        self.assertEqual(xray.calculate_severity_level, 2)

        xray.atelectasis = 0.80
        self.assertEqual(xray.calculate_severity_level, 1)

class ModelLoaderTests(SimpleTestCase):
    def setUp(self):
        clear_model_cache()

    @patch('xrayapp.model_loader.xrv')
    @patch('xrayapp.model_loader.torch')
    def test_load_model_caching(self, mock_torch, mock_xrv):
        # Mock the model creation
        mock_model = MagicMock()
        mock_xrv.models.DenseNet.return_value = mock_model
        
        # Load model first time
        model1, resize1 = load_model('densenet')
        
        # Verify it was created
        mock_xrv.models.DenseNet.assert_called_once()
        self.assertEqual(model1, mock_model)
        self.assertEqual(resize1, 224)
        
        # Load model second time
        model2, resize2 = load_model('densenet')
        
        # Verify it wasn't created again (cached)
        mock_xrv.models.DenseNet.assert_called_once()
        self.assertEqual(model2, model1)
    
    @patch('xrayapp.model_loader.xrv')
    @patch('xrayapp.model_loader.torch')
    def test_load_resnet(self, mock_torch, mock_xrv):
        mock_model = MagicMock()
        mock_xrv.models.ResNet.return_value = mock_model
        
        model, resize = load_model('resnet')
        
        mock_xrv.models.ResNet.assert_called_once()
        self.assertEqual(model, mock_model)
        self.assertEqual(resize, 512)
