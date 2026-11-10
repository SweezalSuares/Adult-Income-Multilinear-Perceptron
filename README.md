# Income Prediction with an MLP: Streamlit App

Interactive Streamlit version of **Lab 6** (`Sweezal_Suares_547_lab6.ipynb`): a Keras Multi-Layer Perceptron that predicts
whether a person's income is `>50K` or `<=50K` using the Adult (Census Income) dataset.

## Project structure

```
income_mlp_streamlit/
├── app.py                              # Streamlit application
├── Sweezal_Suares_547_lab6.ipynb       # Original notebook
├── requirements.txt                    # Python dependencies
├── data/
│   └── README.md                       # Put adult_train.csv here (optional)
├── .streamlit/config.toml              # Theme / upload size
├── .gitignore
└── README.md
```

## Setup (Windows)

```powershell
cd income_mlp_streamlit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow supports Python 3.9 to 3.12 (check the TensorFlow install guide for newer versions).

## Run

```powershell
streamlit run app.py
```

Then either upload `adult_train.csv` from the sidebar, or copy it to `data/adult_train.csv`.
The dataset is not bundled because it was not included with the notebook.

## What the app does

| Tab | Content |
|---|---|
| **Data** | Preview, target balance, missing values, per-feature explorer, preprocessing summary |
| **Training** | Train/validation accuracy and loss curves, overfitting hint, layer-by-layer parameter count |
| **Evaluation** | Accuracy / precision / recall / F1 for each activation, confusion matrix, classification report, MAE/MSE/RMSE/R² on probabilities, adjustable threshold |
| **Predict** | Fill a form for one person and get the predicted income class and probability |
| **About** | Pipeline summary |

Sidebar controls: activation function(s) (ReLU, Tanh, Sigmoid, ELU), hidden layer sizes, epochs, batch size,
early stopping, test size and random seed. Defaults match the notebook (64 → 32 → 1, 20 epochs, batch 32).

## Notes

- **Leakage fix:** the notebook fits the preprocessor on the whole dataset before the train/test split. The app splits
  first and fits the preprocessor on the training portion only, so numbers will differ slightly from the notebook.
- In the notebook run, validation loss was lowest around epoch 4 and rose afterwards, so enabling **Early stopping**
  is a good thing to try.
- Whitespace around categorical values is stripped and `?` is treated as missing.

## Author

Sweezal Suares · M.Sc. AI & ML, Christ University, Bangalore
