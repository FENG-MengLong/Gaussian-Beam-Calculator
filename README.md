# Gaussian Beam Calculator

A Streamlit web application with three tools:

1. Fit measured beam radii to obtain waist radius, waist position, M², Rayleigh range, and divergence.
2. Calculate and plot propagation through a thin lens.
3. Scan lens position and export the calculated results.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally <http://localhost:8501>.

## Fit data format

Upload a CSV containing `z` and `w` columns. Select their units in the interface. `w` is the **beam radius**, not diameter, and is the 1/e² intensity radius.

```csv
z,w
-200,0.640
-100,0.540
0,0.500
100,0.540
200,0.640
```

## Deployment

Push `app.py` and `requirements.txt` to GitHub and create an app at [Streamlit Community Cloud](https://share.streamlit.io/). Set the entry point to `app.py`.

## Model limitations

The calculations assume paraxial propagation, a thin ideal lens, and a constant M². Fit accuracy depends strongly on measurements spanning both sides of the waist and a sufficiently large propagation range.
