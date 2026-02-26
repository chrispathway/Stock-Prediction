# Stock prediction

Predicts next-day stock returns using a Random Forest (moving averages, price vs MA, daily returns). Compares buy-and-hold vs “only invest when the model says up” and plots the result. Optional: saves an animated wealth curve as a GIF.

**Default:** SPY (S&P 500), 10 years of data, 80% train / 20% validation.

## Setup

```bash
cd Stock-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Plots show up on screen and it writes `wealth_animation.gif` in the project folder. To change the ticker or years, edit the config at the top of `main.py`.
