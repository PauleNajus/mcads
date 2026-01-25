# Multi-label Chest Abnormality Detection System (MCADS)

A Django-based web application for automated chest X-ray analysis using deep learning models. MCADS provides healthcare professionals with AI-powered tools to detect multiple pathologies in chest X-rays with interpretability features.

## Features

- **Multi-label Classification**: Detects 18 different chest abnormalities including:
  - Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
  - Emphysema, Enlarged Cardiomediastinum, Fibrosis, Fracture
  - Hernia, Infiltration, Lung Lesion, Lung Opacity, Mass
  - Nodule, Pleural Thickening, Pneumonia, Pneumothorax

- **User Management**: Secure authentication system with user profiles and hospital affiliations
- **Image Upload & Analysis**: Support for various image formats with preprocessing
- **Prediction History**: Track and review past analyses
- **Interpretability**: Visualizations to understand model predictions
- **Responsive UI**: Bootstrap-based interface optimized for medical workflows

## X & Y

- X (input) --> chest X-ray
- Y (output) --> 18 per-class pathologies probabilities and GRAD-CAM heatmap.

Example of a GRAD-CAM heatmap:

![GRAD-CAM heatmap example](https://github.com/user-attachments/assets/f9f422bb-3954-4290-9320-b3e2dfd529bb)

## Technology Stack

- **Backend**: Django 5.2, Python 3.11.9
- **Machine Learning**: PyTorch 2.7.0, TorchXRayVision
- **Database**: PostgreSQL (recommended: 15+)
- **Frontend**: Django-Bootstrap5 25.1, HTML5, CSS3, JavaScript
- **Deployment**: Cross-platform (Windows/Linux)

## Installation

### Prerequisites

- Python 3.11.9
- pip (Python package manager)
- Git
- PostgreSQL (recommended: 15+)

### Setup Instructions

1. **Clone the repository:**

```bash
git clone <repository-url>
cd mcads
```

1. **Create and activate virtual environment:**

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

1. **Set up the database:**

```bash
python manage.py migrate
```

1. **Create a superuser:**

```bash
python manage.py createsuperuser
```

1. **Collect static files:**

```bash
python manage.py collectstatic --noinput
```

1. **Run the development server:**

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Usage

### Web Interface

1. **Login/Register**: Create an account or log in with existing credentials
2. **Upload Image**: Navigate to the analysis page and upload a chest X-ray image
3. **View Results**: Review prediction scores for all pathologies
4. **Interpretability**: View visualizations to understand model decisions
5. **History**: Access previous analyses and results

## Project Structure

```text
mcads/
├── mcads_project/          # Django project configuration
├── xrayapp/               # Main application
│   ├── models.py          # Database models
│   ├── views.py           # View controllers
│   ├── forms.py           # Form definitions
│   ├── interpretability.py # Interpretability implementation
│   ├── utils.py           # Utility functions
│   └── templates/         # HTML templates
├── static/                # Static files (CSS, JS, images)
├── media/                 # User uploaded files
├── tests/                 # Test files and sample images
└── requirements.txt       # Python dependencies
```

## Models

MCADS uses pre-trained models from the TorchXRayVision library:

- **DenseNet-121**: Primary model for multi-label classification (all 18 pathologies)
- **ResNet-50**: Alternative model with filtered pathology set (16 pathologies)
- **Input Size**: 224x224 (DenseNet) or 512x512 (ResNet) pixels
- **Preprocessing**: Automatic normalization and resizing
- **Output**: Probability scores for up to 18 pathologies

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mcads_db
DB_USER=mcads_user
DB_PASSWORD=your-db-password
```

### Settings

Key configuration options in `mcads_project/settings.py`:

- `MEDIA_ROOT`: Directory for uploaded images
- `STATIC_ROOT`: Directory for static files
- `LOGGING`: Logging configuration
- `DATABASES`: Database configuration

## Testing

Run the test suite:

```bash
python manage.py test
```

### Sample Images

Test images are available in the `tests/` directory:

- `normal.jpeg`: Normal chest X-ray
- `viral_pneumonia.jpeg`: Pneumonia case
- `00000001_000.png`: Sample from NIH dataset

## Deployment

### Production Considerations

1. **Database**: PostgreSQL (required)
2. **Static Files**: Use a CDN or reverse proxy for static file serving
3. **Security**: Update `SECRET_KEY` and disable `DEBUG`
4. **WSGI Server**: Use Gunicorn or uWSGI
5. **Reverse Proxy**: Nginx or Apache for production

### Docker Deployment

```dockerfile
FROM python:3.11.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

## Acknowledgments

- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) for the pre-trained models
- Django and PyTorch communities for excellent documentation
- Healthcare professionals for domain expertise

## Disclaimer

This software is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.
