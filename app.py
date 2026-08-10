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
LENGTH_TO_MM = {"nm": 1e-6, "µm": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1e3}
WAVELENGTH_UNITS = ["nm", "µm", "mm", "m"]
GEOMETRY_UNITS = ["µm", "mm", "cm", "m"]
INPUT_UNIT_DEFAULTS = {
    "lens_lambda_unit": "nm",
    "lens_w0_unit": "mm",
    "lens_s_unit": "mm",
    "lens_f_unit": "mm",
    "lens_plot_min_unit": "mm",
    "lens_plot_max_unit": "mm",
    "scan_lambda_unit": "nm",
    "scan_w0_unit": "mm",
    "scan_f_unit": "mm",
    "scan_min_unit": "mm",
    "scan_max_unit": "mm",
}


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


def sync_value_forward(source_key, *target_keys):
    """Copy an edited value into widgets in the following tabs."""
    for target_key in target_keys:
        st.session_state[target_key] = st.session_state[source_key]


def sync_length_forward(source_key, unit_key, target_key, target_unit_key):
    """Copy a length and its selected unit into the following tab."""
    source_unit = st.session_state[unit_key]
    st.session_state[target_unit_key] = source_unit
    st.session_state[f"_previous_{target_unit_key}"] = source_unit
    st.session_state[target_key] = st.session_state[source_key]


def sync_fit_wavelength_forward():
    """Pass tab-1 wavelength and its fixed nm unit forward."""
    wavelength_nm = st.session_state["fit_lambda"]
    for value_key, unit_key in (
        ("lens_lambda_input", "lens_lambda_unit"),
        ("scan_lambda_input", "scan_lambda_unit"),
    ):
        st.session_state[unit_key] = "nm"
        st.session_state[f"_previous_{unit_key}"] = "nm"
        st.session_state[value_key] = wavelength_nm


def convert_length_input_unit(
    value_key, unit_key, target_key=None, target_unit_key=None
):
    """Change one input's display unit without changing its physical value."""
    previous_key = f"_previous_{unit_key}"
    old_unit = st.session_state[previous_key]
    new_unit = st.session_state[unit_key]
    if value_key in st.session_state:
        st.session_state[value_key] *= LENGTH_TO_MM[old_unit] / LENGTH_TO_MM[new_unit]
    st.session_state[previous_key] = new_unit
    if target_key is not None and value_key in st.session_state:
        sync_length_forward(value_key, unit_key, target_key, target_unit_key)


for unit_key, default_unit in INPUT_UNIT_DEFAULTS.items():
    st.session_state.setdefault(
        f"_previous_{unit_key}",
        st.session_state.get(unit_key, default_unit),
    )

st.title("Gaussian Beam Calculator")
st.caption("Beam radii are 1/e² intensity radii. Select units beside each length input.")

fit_tab, transform_tab, scan_tab = st.tabs(
    ["1 · Fit measured beam", "2 · Thin-lens transformation", "3 · Scan lens position"]
)

with fit_tab:
    st.subheader("Fit $w_0$, $z_0$, and $M^2$ from measurements")
    left, right = st.columns([1, 2])
    with left:
        fit_wavelength_nm = st.number_input(
            "Wavelength [nm]",
            min_value=1.0,
            value=780.0,
            key="fit_lambda",
            on_change=sync_fit_wavelength_forward,
        )
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
        edited = st.data_editor(input_data, num_rows="dynamic", width="stretch", key="fit_data")

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
    st.caption("Each length input has an independent unit. Results and plot axes are shown in mm.")

    wavelength_value_col, wavelength_unit_col, waist_value_col, waist_unit_col, m2_col = st.columns(
        [2, 1, 2, 1, 2]
    )
    wavelength_unit = wavelength_unit_col.selectbox(
        "Wavelength unit",
        WAVELENGTH_UNITS,
        key="lens_lambda_unit",
        on_change=convert_length_input_unit,
        args=("lens_lambda_input", "lens_lambda_unit", "scan_lambda_input", "scan_lambda_unit"),
    )
    wavelength_input = wavelength_value_col.number_input(
        f"Wavelength [{wavelength_unit}]",
        min_value=LENGTH_TO_MM["nm"] / LENGTH_TO_MM[wavelength_unit],
        value=780.0 * LENGTH_TO_MM["nm"] / LENGTH_TO_MM[wavelength_unit],
        step=LENGTH_TO_MM["nm"] / LENGTH_TO_MM[wavelength_unit],
        key="lens_lambda_input",
        on_change=sync_length_forward,
        args=("lens_lambda_input", "lens_lambda_unit", "scan_lambda_input", "scan_lambda_unit"),
    )
    w0_unit = waist_unit_col.selectbox(
        "Waist unit",
        GEOMETRY_UNITS,
        index=1,
        key="lens_w0_unit",
        on_change=convert_length_input_unit,
        args=("lens_w0_input", "lens_w0_unit", "scan_w0_input", "scan_w0_unit"),
    )
    w0_input = waist_value_col.number_input(
        f"Input waist radius [{w0_unit}]",
        min_value=1e-6 / LENGTH_TO_MM[w0_unit],
        value=0.5 / LENGTH_TO_MM[w0_unit],
        step=0.1 / LENGTH_TO_MM[w0_unit],
        key="lens_w0_input",
        on_change=sync_length_forward,
        args=("lens_w0_input", "lens_w0_unit", "scan_w0_input", "scan_w0_unit"),
    )
    m2_lens = m2_col.number_input(
        "M²",
        min_value=1.0,
        value=1.0,
        step=0.1,
        key="lens_m2",
        on_change=sync_value_forward,
        args=("lens_m2", "scan_m2"),
    )

    s_value_col, s_unit_col, f_value_col, f_unit_col = st.columns([2, 1, 2, 1])
    s_unit = s_unit_col.selectbox(
        "Waist → lens unit",
        GEOMETRY_UNITS,
        index=1,
        key="lens_s_unit",
        on_change=convert_length_input_unit,
        args=("lens_s_input", "lens_s_unit"),
    )
    s_input = s_value_col.number_input(
        f"Waist → lens distance [{s_unit}]",
        value=150.0 / LENGTH_TO_MM[s_unit],
        step=1.0 / LENGTH_TO_MM[s_unit],
        key="lens_s_input",
    )
    f_unit = f_unit_col.selectbox(
        "Focal-length unit",
        GEOMETRY_UNITS,
        index=1,
        key="lens_f_unit",
        on_change=convert_length_input_unit,
        args=("lens_f_input", "lens_f_unit", "scan_f_input", "scan_f_unit"),
    )
    f_input = f_value_col.number_input(
        f"Focal length [{f_unit}]",
        value=100.0 / LENGTH_TO_MM[f_unit],
        step=1.0 / LENGTH_TO_MM[f_unit],
        key="lens_f_input",
        on_change=sync_length_forward,
        args=("lens_f_input", "lens_f_unit", "scan_f_input", "scan_f_unit"),
    )

    min_value_col, min_unit_col, max_value_col, max_unit_col = st.columns([2, 1, 2, 1])
    plot_min_unit = min_unit_col.selectbox(
        "Plot-minimum unit",
        GEOMETRY_UNITS,
        index=1,
        key="lens_plot_min_unit",
        on_change=convert_length_input_unit,
        args=("lens_plot_min", "lens_plot_min_unit"),
    )
    plot_min_input = min_value_col.number_input(
        f"Plot region minimum [{plot_min_unit}]",
        value=-500.0 / LENGTH_TO_MM[plot_min_unit],
        step=10.0 / LENGTH_TO_MM[plot_min_unit],
        help="Absolute propagation position; the input waist is at z = 0.",
        key="lens_plot_min",
    )
    plot_max_unit = max_unit_col.selectbox(
        "Plot-maximum unit",
        GEOMETRY_UNITS,
        index=1,
        key="lens_plot_max_unit",
        on_change=convert_length_input_unit,
        args=("lens_plot_max", "lens_plot_max_unit"),
    )
    plot_max_input = max_value_col.number_input(
        f"Plot region maximum [{plot_max_unit}]",
        value=650.0 / LENGTH_TO_MM[plot_max_unit],
        step=10.0 / LENGTH_TO_MM[plot_max_unit],
        help="Absolute propagation position; the lens is at z = s.",
        key="lens_plot_max",
    )

    wavelength_mm = wavelength_input * LENGTH_TO_MM[wavelength_unit]
    w0_mm = w0_input * LENGTH_TO_MM[w0_unit]
    s_mm = s_input * LENGTH_TO_MM[s_unit]
    f_mm = f_input * LENGTH_TO_MM[f_unit]
    plot_min_mm = plot_min_input * LENGTH_TO_MM[plot_min_unit]
    plot_max_mm = plot_max_input * LENGTH_TO_MM[plot_max_unit]

    if abs(f_mm) < 1e-12:
        st.error("Focal length cannot be zero.")
    elif plot_max_mm <= plot_min_mm:
        st.error("Plot region maximum must exceed the minimum.")
    else:
        z_r, s_prime, z_r_prime, w0_prime, waist_position = lens_transform(
            wavelength_mm, w0_mm, m2_lens, s_mm, f_mm
        )
        values = st.columns(5)
        with values[0]: metric("Input Rayleigh range", z_r, "mm")
        with values[1]: metric("Lens → new waist", s_prime, "mm")
        with values[2]: metric("New waist position", waist_position, "mm")
        with values[3]: metric("Output Rayleigh range", z_r_prime, "mm")
        with values[4]: metric("Output waist radius", w0_prime, "mm")

        before_start = plot_min_mm
        before_stop = min(s_mm, plot_max_mm)
        after_start = max(s_mm, plot_min_mm)
        after_stop = plot_max_mm

        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        if before_stop > before_start:
            before = np.linspace(before_start, before_stop, 700)
            w_before = w0_mm * np.sqrt(1 + (before / z_r) ** 2)
            line, = ax.plot(before, w_before, label="Incident beam")
            ax.plot(before, -w_before, color=line.get_color())

        if after_stop > after_start:
            after = np.linspace(after_start, after_stop, 700)
            w_after = w0_prime * np.sqrt(1 + ((after - waist_position) / z_r_prime) ** 2)
            w_free = w0_mm * np.sqrt(1 + (after / z_r) ** 2)
            for y, label, style, alpha in [
                (w_after, "After lens", "-", 1),
                (w_free, "Without lens", "--", 0.5),
            ]:
                line, = ax.plot(after, y, style, alpha=alpha, label=label)
                ax.plot(after, -y, style, alpha=alpha, color=line.get_color())

        ax.axvline(s_mm, ls=":", color="black", label="Lens")
        ax.axvline(0, ls=":", alpha=0.25)
        ax.scatter(
            [0, waist_position],
            [0, 0],
            s=12,
            linewidths=0.4,
            color="black",
            zorder=4,
        )
        ax.set(
            xlabel="Propagation position z [mm]",
            ylabel="Beam radius ±w(z) [mm]",
            title="Gaussian Beam Through a Thin Lens",
            xlim=(plot_min_mm, plot_max_mm),
        )
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig, width="content")

with scan_tab:
    st.subheader("Scan the lens position")
    st.caption("Each length input has an independent unit. Plot axes and downloaded data use mm.")

    lambda_value_col, lambda_unit_col, w0_value_col, w0_unit_col, m2_col = st.columns(
        [2, 1, 2, 1, 2]
    )
    scan_lambda_unit = lambda_unit_col.selectbox(
        "Wavelength unit",
        WAVELENGTH_UNITS,
        key="scan_lambda_unit",
        on_change=convert_length_input_unit,
        args=("scan_lambda_input", "scan_lambda_unit"),
    )
    scan_lambda_input = lambda_value_col.number_input(
        f"Wavelength [{scan_lambda_unit}]",
        min_value=LENGTH_TO_MM["nm"] / LENGTH_TO_MM[scan_lambda_unit],
        value=780.0 * LENGTH_TO_MM["nm"] / LENGTH_TO_MM[scan_lambda_unit],
        step=LENGTH_TO_MM["nm"] / LENGTH_TO_MM[scan_lambda_unit],
        key="scan_lambda_input",
    )
    scan_w0_unit = w0_unit_col.selectbox(
        "Waist unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_w0_unit",
        on_change=convert_length_input_unit,
        args=("scan_w0_input", "scan_w0_unit"),
    )
    scan_w0_input = w0_value_col.number_input(
        f"Input waist radius [{scan_w0_unit}]",
        min_value=1e-6 / LENGTH_TO_MM[scan_w0_unit],
        value=0.5 / LENGTH_TO_MM[scan_w0_unit],
        step=0.1 / LENGTH_TO_MM[scan_w0_unit],
        key="scan_w0_input",
    )
    scan_m2 = m2_col.number_input(
        "M²",
        min_value=1.0,
        value=1.0,
        step=0.1,
        key="scan_m2",
    )

    f_value_col, f_unit_col, min_value_col, min_unit_col, max_value_col, max_unit_col = st.columns(
        [2, 1, 2, 1, 2, 1]
    )
    scan_f_unit = f_unit_col.selectbox(
        "Focal-length unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_f_unit",
        on_change=convert_length_input_unit,
        args=("scan_f_input", "scan_f_unit"),
    )
    scan_f_input = f_value_col.number_input(
        f"Focal length [{scan_f_unit}]",
        value=100.0 / LENGTH_TO_MM[scan_f_unit],
        step=1.0 / LENGTH_TO_MM[scan_f_unit],
        key="scan_f_input",
    )
    scan_min_unit = min_unit_col.selectbox(
        "Minimum-position unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_min_unit",
        on_change=convert_length_input_unit,
        args=("scan_min_input", "scan_min_unit"),
    )
    scan_min_input = min_value_col.number_input(
        f"Minimum lens position [{scan_min_unit}]",
        value=-500.0 / LENGTH_TO_MM[scan_min_unit],
        step=10.0 / LENGTH_TO_MM[scan_min_unit],
        key="scan_min_input",
    )
    scan_max_unit = max_unit_col.selectbox(
        "Maximum-position unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_max_unit",
        on_change=convert_length_input_unit,
        args=("scan_max_input", "scan_max_unit"),
    )
    scan_max_input = max_value_col.number_input(
        f"Maximum lens position [{scan_max_unit}]",
        value=1000.0 / LENGTH_TO_MM[scan_max_unit],
        step=10.0 / LENGTH_TO_MM[scan_max_unit],
        key="scan_max_input",
    )

    scan_lambda_mm = scan_lambda_input * LENGTH_TO_MM[scan_lambda_unit]
    scan_w0_mm = scan_w0_input * LENGTH_TO_MM[scan_w0_unit]
    scan_f_mm = scan_f_input * LENGTH_TO_MM[scan_f_unit]
    scan_min_mm = scan_min_input * LENGTH_TO_MM[scan_min_unit]
    scan_max_mm = scan_max_input * LENGTH_TO_MM[scan_max_unit]

    if scan_max_mm <= scan_min_mm or abs(scan_f_mm) < 1e-12:
        st.error("Maximum position must exceed minimum position, and focal length cannot be zero.")
    else:
        positions = np.linspace(scan_min_mm, scan_max_mm, 1200)
        scan_zr = np.pi * scan_w0_mm**2 / (scan_m2 * scan_lambda_mm)
        q_in = positions + 1j * scan_zr
        q_out = 1 / (1 / q_in - 1 / scan_f_mm)
        separations = -q_out.real
        output_zr = q_out.imag
        output_w0 = np.sqrt(scan_m2 * scan_lambda_mm * output_zr / np.pi)
        global_waists = positions + separations

        fig, axes = plt.subplots(3, 1, figsize=(4.0, 3.4), sharex=True)
        axes[0].plot(positions, global_waists)
        axes[0].set_ylabel(r"$z_{\mathrm{waist}}$ [mm]")
        axes[0].set_title("Transformation vs lens position")
        axes[1].plot(positions, separations)
        axes[1].axhline(scan_f_mm, ls="--", alpha=0.5, label="Focal length")
        axes[1].set_ylabel(r"$s'$ [mm]")
        axes[1].legend()
        axes[2].plot(positions, output_w0)
        axes[2].set(ylabel=r"$w_0'$ [mm]", xlabel=r"Lens position $s$ [mm]")
        for axis in axes:
            axis.grid(alpha=0.3)
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
