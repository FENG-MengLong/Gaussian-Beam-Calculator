# Gaussian Beam Calculator

A Streamlit application for fitting Gaussian-beam measurements and calculating beam propagation through as many as four thin lenses.

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

Shared parameters and their selected units flow forward through the tabs: tab 1 passes only the wavelength in nanometres to tabs 2 and 3, while tab 2 passes wavelength, waist radius, and $M^2$ to tab 3. Fitted results from tab 1 remain local to tab 1.

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

## 2. Multi-lens transformation

Use this tab to calculate how a Gaussian beam changes after passing through one to four thin lenses in sequence.

Enter:

- Wavelength and its unit
- Input waist radius and its unit
- Beam-quality factor $M^2$
- Number of lenses (up to four)
- Absolute position and focal length for each lens, with independent units
- Minimum and maximum propagation positions, each with its own unit

The input waist is defined as position $z=0$. Lens positions must increase in the beam-propagation direction. The application calculates the output after every lens and summarizes the final beam with:

- Input Rayleigh range
- Distance from the final lens to the final waist
- Global position of the final waist
- Final Rayleigh range
- Final waist radius

The plot shows the incident beam and each sequentially transformed segment. A table beside the plot lists sampled final-beam radii against position measured from the last lens. Beneath the table, enter any non-negative relative position to calculate its exact beam radius. Calculated length results and plot axes are displayed in millimetres.

## 3. Single lens scan

Use this tab to find a positive thin lens that focuses the incident beam at a known target plane.

1. Enter the wavelength, input waist radius, $M^2$, and target position measured from the incident waist at $z=0$.
2. Enter the lens-position scan range. The entire range must be between the incident waist and target plane.
3. Enter the positive focal-length scan range.
4. Read the 2D map: its color gives the beam radius at the target for every lens-position and focal-length pair. A logarithmic or linear color scale can be selected.
5. Use the overlaid contours to assess the target plane relative to the transformed waist:
   - White: $|z_{target}-z_{waist}|=z_R'/2$
   - Orange: $|z_{target}-z_{waist}|=z_R'$
6. Select a fixed lens position or focal length to draw the corresponding vertical or horizontal slice on the map. The compact slice plot shows the target beam radius along the other scan axis and marks its minimum.
7. Use **Download scan data** to export the map values and transformed-beam parameters as a CSV file.

## Model assumptions

The calculations assume:

- Paraxial Gaussian-beam propagation
- Thin, ideal lenses
- A constant $M^2$
- No aberrations, clipping, or astigmatism

Check all units and sign conventions before using the results in an experiment.
