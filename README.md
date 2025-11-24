# amazon-bestseller-ml

Structure:

- data/books.csv      # put your dataset here when instructed
- models/             # saved models will go here
- venv/               # virtual environment
- notebook.ipynb      # notebook for EDA/training
- app.py              # script/Flask app for prediction
- requirements.txt

Workflow:
1. Use notebook.ipynb to explore, clean, train the model and save joblib files into models/.
2. Use app.py to load saved model(s) for predictions or to run a simple prediction API.
