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
- **RESTful API**: Programmatic access to prediction services

## Technology Stack

- **Backend**: Django 5.2, Python 3.11.9
- **Machine Learning**: PyTorch 2.7.0, TorchXRayVision
- **Database**: SQLite 3.49.1
- **Frontend**: Django-Bootstrap5 25.1, HTML5, CSS3, JavaScript
- **Deployment**: Cross-platform (Windows/Linux)

## Installation

### Prerequisites

- Python 3.11.9
- pip (Python package manager)
- Git

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

### API Usage

The application provides REST endpoints for programmatic access:

```python
import requests

# Upload and analyze image
with open('chest_xray.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze/',
        files={'image': f},
        headers={'Authorization': 'Bearer your-token'}
    )

results = response.json()
```

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

- **ResNet-50**: Primary model for multi-label classification
- **Input Size**: 224x224 or 512x512 pixels
- **Preprocessing**: Automatic normalization and resizing
- **Output**: Probability scores for 18 pathologies

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
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

1. **Database**: Consider PostgreSQL for production
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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) for the pre-trained models
- Django and PyTorch communities for excellent documentation
- Healthcare professionals for domain expertise

## Support

For issues and questions:

- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the test cases for usage examples

## Disclaimer

This software is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.
