# Gaussian Beam Calculator

A Streamlit application for fitting Gaussian-beam measurements and calculating beam propagation through a thin lens.

## Start the application

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

Run the application from the project directory:

```bash
python3 -m streamlit run app.py
```

Open the URL displayed in the terminal, normally <http://localhost:8501>.

> All beam sizes in this application are **1/e² intensity radii**, not diameters.

## 1. Fit measured beam

Use this tab to determine the beam-waist radius, waist position, and beam-quality factor from measured beam radii.

1. Enter the laser wavelength in nanometres.
2. Select the units used by the distance (`z`) and radius (`w`) columns.
3. Upload a CSV file or edit the example data directly in the table.
4. Click **Fit beam**.

The application reports:

- Waist radius, $w_0$
- Waist position, $z_0$
- Beam-quality factor, $M^2$
- Rayleigh range, $z_R$
- Far-field divergence half-angle
- Coefficient of determination, $R^2$
- One-standard-deviation ($1\sigma$) uncertainties for the fitted parameters

The fitted curve and measured points are plotted together. Use **Download fit result** to save the calculated values as a CSV file.

### Measurement CSV format

The CSV file must contain columns named `z` and `w`:

```csv
z,w
-200,0.640
-100,0.540
0,0.500
100,0.540
200,0.640
```

For a reliable fit:

- Use radius rather than diameter.
- Include measurements on both sides of the waist when possible.
- Measure over a sufficiently large propagation range.
- Use at least four valid measurement points.

## 2. Thin-lens transformation

Use this tab to calculate how a Gaussian beam changes after passing through a thin lens.

Enter:

- Wavelength in nanometres
- Input waist radius in millimetres
- Beam-quality factor $M^2$
- Distance from the input waist to the lens in millimetres
- Lens focal length in millimetres

The input waist is defined as position $z=0$, and the lens is located at $z=s$. The application calculates:

- Input Rayleigh range
- Distance from the lens to the new waist
- Global position of the new waist
- Output Rayleigh range
- Output waist radius

The plot compares the incident beam, transformed beam, and free-propagating beam without a lens.

## 3. Scan lens position

Use this tab to examine how the output beam changes as the lens moves.

1. Enter the wavelength, input waist radius, $M^2$, and focal length.
2. Enter the minimum and maximum lens positions.
3. View the calculated curves for:
   - Output-waist position
   - Lens-to-waist separation
   - Output-waist radius
4. Use **Download scan data** to export all calculated scan points as a CSV file.

## Model assumptions

The calculations assume:

- Paraxial Gaussian-beam propagation
- A thin, ideal lens
- A constant $M^2$
- No aberrations, clipping, or astigmatism

Check all units and sign conventions before using the results in an experiment.
