import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

st.set_page_config(page_title="Gaussian Beam Calculator", page_icon="🔦", layout="wide")

# Compact typography for plots embedded in the web page.
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 0.8,
    "grid.linewidth": 0.4,
})

NM_TO_M = 1e-9
MM_TO_M = 1e-3


def gaussian_beam_radius(z, w0, z0, m2, wavelength_m):
    """Return the 1/e² intensity radius in metres."""
    return w0 * np.sqrt(
        1.0 + (m2 * wavelength_m * (z - z0) / (np.pi * w0**2)) ** 2
    )


def fit_gaussian_beam(z, w, wavelength_m):
    order = np.argsort(z)
    z, w = np.asarray(z, float)[order], np.asarray(w, float)[order]
    if len(z) < 4:
        raise ValueError("At least four valid measurements are required.")
    if np.ptp(z) == 0 or np.any(w <= 0):
        raise ValueError("Distances must span a range and all radii must be positive.")

    p0 = [float(np.min(w)), float(z[np.argmin(w)]), 1.2]
    span = np.ptp(z)
    bounds = ([1e-12, np.min(z) - 10 * span, 1.0], [np.inf, np.max(z) + 10 * span, 100.0])

    def model(position, w0, z0, m2):
        return gaussian_beam_radius(position, w0, z0, m2, wavelength_m)

    popt, pcov = curve_fit(model, z, w, p0=p0, bounds=bounds, maxfev=30000)
    errors = np.sqrt(np.diag(pcov))
    w0, z0, m2 = popt
    z_rayleigh = np.pi * w0**2 / (m2 * wavelength_m)
    theta = m2 * wavelength_m / (np.pi * w0)
    residuals = w - model(z, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((w - np.mean(w)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return popt, errors, pcov, z_rayleigh, theta, r_squared, model


def lens_transform(wavelength_mm, w0, m2, s, focal_length):
    z_rayleigh = np.pi * w0**2 / (m2 * wavelength_mm)
    q_in = s + 1j * z_rayleigh
    q_out = 1 / (1 / q_in - 1 / focal_length)
    s_prime = -q_out.real
    z_rayleigh_prime = q_out.imag
    w0_prime = np.sqrt(m2 * wavelength_mm * z_rayleigh_prime / np.pi)
    return z_rayleigh, s_prime, z_rayleigh_prime, w0_prime, s + s_prime


def metric(label, value, unit="", error=None):
    if error is None:
        display = f"{value:.6g}"
    else:
        display = f"{value:.6g} ± {error:.3g}"
    st.markdown(f"**{label}**  \n{display} {unit}".strip())


st.title("Gaussian Beam Calculator")
st.caption("Beam radii are 1/e² intensity radii. Wavelength inputs are in nm.")

fit_tab, transform_tab, scan_tab = st.tabs(
    ["1 · Fit measured beam", "2 · Thin-lens transformation", "3 · Scan lens position"]
)

with fit_tab:
    st.subheader("Fit $w_0$, $z_0$, and $M^2$ from measurements")
    left, right = st.columns([1, 2])
    with left:
        fit_wavelength_nm = st.number_input("Wavelength [nm]", min_value=1.0, value=780.0, key="fit_lambda")
        z_unit = st.selectbox("Distance column unit", ["m", "mm", "in"], index=1)
        w_unit = st.selectbox("Radius column unit", ["m", "mm", "µm"], index=1)
        uploaded = st.file_uploader(
            "CSV file with columns z and w", type="csv", help="w must be radius, not diameter."
        )
        st.download_button(
            "Download example CSV",
            data="z,w\n-200,0.640\n-100,0.540\n0,0.500\n100,0.540\n200,0.640\n",
            file_name="beam_measurements.csv",
            mime="text/csv",
        )

    default_data = pd.DataFrame(
        {"z": [-300, -200, -100, 0, 100, 200, 300], "w": [0.775, 0.635, 0.535, 0.500, 0.535, 0.635, 0.775]}
    )
    if uploaded is not None:
        try:
            input_data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            input_data = default_data
    else:
        input_data = default_data

    with right:
        edited = st.data_editor(input_data, num_rows="dynamic", use_container_width=True, key="fit_data")

    if st.button("Fit beam", type="primary"):
        try:
            if not {"z", "w"}.issubset(edited.columns):
                raise ValueError("The table must contain columns named z and w.")
            clean = edited[["z", "w"]].apply(pd.to_numeric, errors="coerce").dropna()
            z_factors = {"m": 1.0, "mm": MM_TO_M, "in": 0.0254}
            w_factors = {"m": 1.0, "mm": MM_TO_M, "µm": 1e-6}
            z = clean["z"].to_numpy() * z_factors[z_unit]
            w = clean["w"].to_numpy() * w_factors[w_unit]
            wavelength_m = fit_wavelength_nm * NM_TO_M
            popt, perr, pcov, z_r, theta, r2, model = fit_gaussian_beam(z, w, wavelength_m)
            w0, z0, m2 = popt

            z_plot = np.linspace(z.min(), z.max(), 600)
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.scatter(z * 1e3, w * 1e3, s=12, linewidths=0.4, label="Measured", zorder=3)
            ax.plot(z_plot * 1e3, model(z_plot, *popt) * 1e3, label="Gaussian fit")
            ax.axvline(z0 * 1e3, ls="--", color="gray", label="Fitted waist")
            ax.set(xlabel="Position z [mm]", ylabel="Beam radius w [mm]")
            ax.grid(alpha=0.3)
            ax.legend()

            plot_col, results_col = st.columns([1, 1])
            with plot_col:
                st.pyplot(fig, width="content")
            with results_col:
                st.markdown("#### Fitted results")
                result_left, result_right = st.columns(2)
                with result_left:
                    metric("Waist radius", w0 * 1e6, "µm", perr[0] * 1e6)
                    metric("M²", m2, error=perr[2])
                    metric("Rayleigh range", z_r * 1e3, "mm")
                with result_right:
                    metric("Waist position", z0 * 1e3, "mm", perr[1] * 1e3)
                    metric("R²", r2)
                    metric("Half-angle", theta * 1e3, "mrad")

            result_csv = pd.DataFrame([{
                "w0_um": w0 * 1e6, "z0_mm": z0 * 1e3, "M2": m2,
                "zR_mm": z_r * 1e3, "theta_mrad": theta * 1e3, "R_squared": r2,
                "w0_error_um": perr[0] * 1e6, "z0_error_mm": perr[1] * 1e3,
                "M2_error": perr[2],
            }]).to_csv(index=False)
            st.download_button("Download fit result", result_csv, "fit_result.csv", "text/csv")
        except Exception as exc:
            st.error(f"Fit failed: {exc}")

with transform_tab:
    st.subheader("Transform a Gaussian beam through a thin lens")
    c1, c2, c3, c4, c5 = st.columns(5)
    wavelength_nm = c1.number_input("Wavelength [nm]", min_value=1.0, value=780.0, key="lens_lambda")
    w0_mm = c2.number_input("Input waist radius [mm]", min_value=1e-6, value=0.5)
    m2_lens = c3.number_input("M²", min_value=1.0, value=1.0, step=0.1)
    s_mm = c4.number_input("Waist → lens distance [mm]", value=150.0)
    f_mm = c5.number_input("Focal length [mm]", value=100.0)

    if abs(f_mm) < 1e-12:
        st.error("Focal length cannot be zero.")
    else:
        wavelength_mm = wavelength_nm * 1e-6
        z_r, s_prime, z_r_prime, w0_prime, waist_position = lens_transform(
            wavelength_mm, w0_mm, m2_lens, s_mm, f_mm
        )
        values = st.columns(5)
        with values[0]: metric("Input Rayleigh range", z_r, "mm")
        with values[1]: metric("Lens → new waist", s_prime, "mm")
        with values[2]: metric("New waist position", waist_position, "mm")
        with values[3]: metric("Output Rayleigh range", z_r_prime, "mm")
        with values[4]: metric("Output waist radius", w0_prime, "mm")

        z_min = min(-0.5 * z_r, waist_position - 3 * z_r_prime)
        z_max = max(s_mm + 0.5 * z_r, waist_position + 3 * z_r_prime)
        before = np.linspace(z_min, s_mm, 700)
        after = np.linspace(s_mm, z_max, 700)
        w_before = w0_mm * np.sqrt(1 + (before / z_r) ** 2)
        w_after = w0_prime * np.sqrt(1 + ((after - waist_position) / z_r_prime) ** 2)
        w_free = w0_mm * np.sqrt(1 + (after / z_r) ** 2)

        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        for x, y, label, style, alpha in [
            (before, w_before, "Incident beam", "-", 1),
            (after, w_after, "After lens", "-", 1),
            (after, w_free, "Without lens", "--", 0.5),
        ]:
            line, = ax.plot(x, y, style, alpha=alpha, label=label)
            ax.plot(x, -y, style, alpha=alpha, color=line.get_color())
        ax.axvline(s_mm, ls=":", color="black", label="Lens")
        ax.axvline(0, ls=":", alpha=0.25)
        ax.scatter([0, waist_position], [0, 0], s=12, linewidths=0.4, color="black", zorder=4)
        ax.set(xlabel="Propagation position z [mm]", ylabel="Beam radius ±w(z) [mm]", title="Gaussian Beam Through a Thin Lens")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig, width="content")

with scan_tab:
    st.subheader("Scan the lens position")
    a, b, c, d, e, fcol = st.columns(6)
    scan_lambda = a.number_input("Wavelength [nm]", min_value=1.0, value=780.0, key="scan_lambda")
    scan_w0 = b.number_input("Input waist radius [mm]", min_value=1e-6, value=0.5, key="scan_w0")
    scan_m2 = c.number_input("M²", min_value=1.0, value=1.0, step=0.1, key="scan_m2")
    scan_f = d.number_input("Focal length [mm]", value=100.0, key="scan_f")
    scan_min = e.number_input("Minimum lens position [mm]", value=-500.0)
    scan_max = fcol.number_input("Maximum lens position [mm]", value=1000.0)

    if scan_max <= scan_min or abs(scan_f) < 1e-12:
        st.error("Maximum position must exceed minimum position, and focal length cannot be zero.")
    else:
        positions = np.linspace(scan_min, scan_max, 1200)
        scan_zr = np.pi * scan_w0**2 / (scan_m2 * scan_lambda * 1e-6)
        q_in = positions + 1j * scan_zr
        q_out = 1 / (1 / q_in - 1 / scan_f)
        separations = -q_out.real
        output_zr = q_out.imag
        output_w0 = np.sqrt(scan_m2 * scan_lambda * 1e-6 * output_zr / np.pi)
        global_waists = positions + separations

        fig, axes = plt.subplots(3, 1, figsize=(4.0, 3.4), sharex=True)
        axes[0].plot(positions, global_waists)
        axes[0].set_ylabel(r"$z_{\mathrm{waist}}$ [mm]")
        axes[0].set_title("Transformation vs lens position", fontsize=7)
        axes[1].plot(positions, separations)
        axes[1].axhline(scan_f, ls="--", alpha=0.5, label="Focal length")
        axes[1].set_ylabel(r"$s'$ [mm]")
        axes[1].legend(fontsize=5.5)
        axes[2].plot(positions, output_w0)
        axes[2].set(ylabel=r"$w_0'$ [mm]", xlabel=r"Lens position $s$ [mm]")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.tick_params(labelsize=5.5)
            axis.xaxis.label.set_size(6)
            axis.yaxis.label.set_size(6)
        fig.tight_layout(pad=0.6, h_pad=0.7)
        st.pyplot(fig, width="content")

        scan_csv = pd.DataFrame({
            "lens_position_mm": positions,
            "output_waist_position_mm": global_waists,
            "lens_to_waist_mm": separations,
            "output_rayleigh_range_mm": output_zr,
            "output_waist_radius_mm": output_w0,
        }).to_csv(index=False)
        st.download_button("Download scan data", scan_csv, "lens_scan.csv", "text/csv")

st.divider()
st.caption("Model: paraxial thin lens, stigmatic propagation, and constant M². Confirm unit conventions before using results in an experiment.")
