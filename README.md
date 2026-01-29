# Glycosense-Non-Invasive-Glucose-Monitoring

Glycosense-Non-Invasive/
├── data/                   # Raw and processed datasets (CSV/NPY)
├── models/                 # Saved SVR models (.pkl, .joblib) and versioning
├── notebooks/              # Experimental training and EDA (kept separate)
├── src/                    # Main source code
│   ├── __init__.py
│   ├── preprocessing.py    # Signal processing, PAS data cleaning
│   ├── inference.py        # Logic to load model and run predictions
│   ├── utils.py            # Helper functions (logging, data loading)
│   └── calibration.py      # C++ wrappers or calibration logic
├── api/                    # Deployment layer
│   ├── app.py              # FastAPI or Flask entry point
│   └── schemas.py          # Data validation for input readings
├── config/
│   └── settings.yaml       # Hyperparameters and threshold constants
├── tests/                  # Validation scripts for SVR accuracy
├── requirements.txt        # Dependency list
├── .gitignore              # Exclude __pycache__, models, and large data
└── README.md               # Setup and deployment instructions
