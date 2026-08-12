import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
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
    "scan_target_unit": "mm",
    "scan_min_unit": "mm",
    "scan_max_unit": "mm",
    "scan_f_min_unit": "mm",
    "scan_f_max_unit": "mm",
}
for lens_number in range(2, 5):
    INPUT_UNIT_DEFAULTS[f"lens_{lens_number}_s_unit"] = "mm"
    INPUT_UNIT_DEFAULTS[f"lens_{lens_number}_f_unit"] = "mm"


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


def multi_lens_transform(wavelength_mm, w0, m2, lenses):
    """Propagate a Gaussian beam through lenses ordered along the z axis."""
    z_rayleigh = np.pi * w0**2 / (m2 * wavelength_mm)
    q_parameter = 1j * z_rayleigh
    previous_position = 0.0
    results = []

    for position, focal_length in lenses:
        q_at_lens = q_parameter + position - previous_position
        q_parameter = 1 / (1 / q_at_lens - 1 / focal_length)
        output_rayleigh = q_parameter.imag
        waist_position = position - q_parameter.real
        output_waist = np.sqrt(m2 * wavelength_mm * output_rayleigh / np.pi)
        results.append(
            {
                "position": position,
                "focal_length": focal_length,
                "lens_to_waist": -q_parameter.real,
                "waist_position": waist_position,
                "rayleigh_range": output_rayleigh,
                "waist_radius": output_waist,
            }
        )
        previous_position = position

    return z_rayleigh, results


@st.cache_data(max_entries=20)
def single_lens_scan(
    wavelength_mm,
    input_waist_mm,
    m2,
    target_position_mm,
    lens_min_mm,
    lens_max_mm,
    focal_min_mm,
    focal_max_mm,
    samples=241,
):
    """Scan a thin lens and return the Gaussian-beam state at a fixed target."""
    lens_positions = np.linspace(lens_min_mm, lens_max_mm, samples)
    focal_lengths = np.linspace(focal_min_mm, focal_max_mm, samples)
    lens_grid, focal_grid = np.meshgrid(lens_positions, focal_lengths)

    input_rayleigh = np.pi * input_waist_mm**2 / (m2 * wavelength_mm)
    q_at_lens = lens_grid + 1j * input_rayleigh
    q_after_lens = 1 / (1 / q_at_lens - 1 / focal_grid)
    q_at_target = q_after_lens + target_position_mm - lens_grid

    output_rayleigh = q_after_lens.imag
    output_waist_position = lens_grid - q_after_lens.real
    output_waist_radius = np.sqrt(m2 * wavelength_mm * output_rayleigh / np.pi)
    target_radius = np.sqrt(
        m2
        * wavelength_mm
        / np.pi
        * (q_at_target.real**2 + q_at_target.imag**2)
        / q_at_target.imag
    )
    target_offset_in_rayleigh = np.abs(q_at_target.real) / output_rayleigh

    return {
        "lens_positions": lens_positions,
        "focal_lengths": focal_lengths,
        "lens_grid": lens_grid,
        "focal_grid": focal_grid,
        "input_rayleigh": input_rayleigh,
        "output_rayleigh": output_rayleigh,
        "output_waist_position": output_waist_position,
        "output_waist_radius": output_waist_radius,
        "target_radius": target_radius,
        "target_offset_in_rayleigh": target_offset_in_rayleigh,
    }


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
    ["1 · Fit measured beam", "2 · Multi-lens transformation", "3 · Single lens scan"]
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
    st.subheader("Transform a Gaussian beam through thin lenses")
    st.caption(
        "Choose one to four lenses in propagation order. Each length input has an "
        "independent unit; results and plot axes are shown in mm."
    )

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

    lens_count = st.selectbox(
        "Number of lenses",
        options=[1, 2, 3, 4],
        key="lens_count",
        help="The beam passes through the lenses from Lens 1 to Lens 4.",
    )

    lens_inputs = []
    default_positions_mm = [150.0, 300.0, 450.0, 600.0]
    for lens_index in range(1, lens_count + 1):
        if lens_index == 1:
            position_key = "lens_s_input"
            position_unit_key = "lens_s_unit"
            focal_key = "lens_f_input"
            focal_unit_key = "lens_f_unit"
        else:
            position_key = f"lens_{lens_index}_s_input"
            position_unit_key = f"lens_{lens_index}_s_unit"
            focal_key = f"lens_{lens_index}_f_input"
            focal_unit_key = f"lens_{lens_index}_f_unit"

        with st.container(border=True):
            st.markdown(f"**Lens {lens_index}**")
            position_value_col, position_unit_col, focal_value_col, focal_unit_col = st.columns(
                [2, 1, 2, 1]
            )
            position_unit = position_unit_col.selectbox(
                f"Lens {lens_index} position unit",
                GEOMETRY_UNITS,
                index=1,
                key=position_unit_key,
                on_change=convert_length_input_unit,
                args=(position_key, position_unit_key),
            )
            position_input = position_value_col.number_input(
                f"Position from input waist [{position_unit}]",
                value=default_positions_mm[lens_index - 1] / LENGTH_TO_MM[position_unit],
                step=1.0 / LENGTH_TO_MM[position_unit],
                help="Absolute position along the propagation axis; the input waist is at z = 0.",
                key=position_key,
            )
            focal_unit = focal_unit_col.selectbox(
                f"Lens {lens_index} focal-length unit",
                GEOMETRY_UNITS,
                index=1,
                key=focal_unit_key,
                on_change=convert_length_input_unit,
                args=(focal_key, focal_unit_key),
            )
            focal_input = focal_value_col.number_input(
                f"Focal length [{focal_unit}]",
                value=100.0 / LENGTH_TO_MM[focal_unit],
                step=1.0 / LENGTH_TO_MM[focal_unit],
                key=focal_key,
            )
        lens_inputs.append(
            (
                position_input * LENGTH_TO_MM[position_unit],
                focal_input * LENGTH_TO_MM[focal_unit],
            )
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
    plot_min_mm = plot_min_input * LENGTH_TO_MM[plot_min_unit]
    plot_max_mm = plot_max_input * LENGTH_TO_MM[plot_max_unit]

    lens_positions = [position for position, _ in lens_inputs]
    if any(abs(focal_length) < 1e-12 for _, focal_length in lens_inputs):
        st.error("Focal lengths cannot be zero.")
    elif any(
        second_position <= first_position
        for first_position, second_position in zip(lens_positions, lens_positions[1:])
    ):
        st.error("Lens positions must increase from Lens 1 through the last selected lens.")
    elif plot_max_mm <= plot_min_mm:
        st.error("Plot region maximum must exceed the minimum.")
    else:
        z_r, lens_results = multi_lens_transform(
            wavelength_mm, w0_mm, m2_lens, lens_inputs
        )
        final_result = lens_results[-1]
        values = st.columns(5)
        with values[0]: metric("Input Rayleigh range", z_r, "mm")
        with values[1]: metric("Last lens → waist", final_result["lens_to_waist"], "mm")
        with values[2]: metric("Final waist position", final_result["waist_position"], "mm")
        with values[3]: metric("Final Rayleigh range", final_result["rayleigh_range"], "mm")
        with values[4]: metric("Final waist radius", final_result["waist_radius"], "mm")

        result_table = pd.DataFrame(
            [
                {
                    "Lens": lens_index,
                    "Position [mm]": result["position"],
                    "Focal length [mm]": result["focal_length"],
                    "Lens to waist [mm]": result["lens_to_waist"],
                    "Waist position [mm]": result["waist_position"],
                    "Output Rayleigh range [mm]": result["rayleigh_range"],
                    "Output waist radius [mm]": result["waist_radius"],
                }
                for lens_index, result in enumerate(lens_results, start=1)
            ]
        )
        st.dataframe(result_table, hide_index=True, width="stretch")

        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        beam_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        segment_starts = [plot_min_mm] + lens_positions
        segment_stops = lens_positions + [plot_max_mm]
        for segment_index, (segment_start, segment_stop) in enumerate(
            zip(segment_starts, segment_stops)
        ):
            visible_start = max(segment_start, plot_min_mm)
            visible_stop = min(segment_stop, plot_max_mm)
            if visible_stop <= visible_start:
                continue

            positions = np.linspace(visible_start, visible_stop, 700)
            if segment_index == 0:
                radii = w0_mm * np.sqrt(1 + (positions / z_r) ** 2)
                label = "Incident beam"
            else:
                result = lens_results[segment_index - 1]
                radii = result["waist_radius"] * np.sqrt(
                    1
                    + (
                        (positions - result["waist_position"])
                        / result["rayleigh_range"]
                    )
                    ** 2
                )
                label = f"After Lens {segment_index}"
            beam_color = beam_colors[segment_index % len(beam_colors)]
            ax.plot(positions, radii, color=beam_color, label=label)
            ax.plot(positions, -radii, color=beam_color)

        waist_positions = [0] + [result["waist_position"] for result in lens_results]
        for profile_index, waist_position in enumerate(waist_positions):
            ax.scatter(
                waist_position,
                0,
                s=18,
                linewidths=0.4,
                edgecolors="black",
                color=beam_colors[profile_index % len(beam_colors)],
                zorder=4,
            )

        for lens_index, lens_position in enumerate(lens_positions, start=1):
            ax.axvline(
                lens_position,
                ls=":",
                color="black",
                alpha=0.7,
                label="Lenses" if lens_index == 1 else None,
            )
        ax.axvline(0, ls=":", alpha=0.25)
        ax.set(
            xlabel="Propagation position z [mm]",
            ylabel="Beam radius ±w(z) [mm]",
            title="Gaussian beam through thin lenses",
            xlim=(plot_min_mm, plot_max_mm),
        )
        ax.grid(alpha=0.3)
        ax.legend()

        output_distance_max = max(0.0, plot_max_mm - lens_positions[-1])
        output_distances = (
            np.linspace(0.0, output_distance_max, 21)
            if output_distance_max > 0
            else np.array([0.0])
        )
        output_radii = final_result["waist_radius"] * np.sqrt(
            1
            + (
                (
                    lens_positions[-1]
                    + output_distances
                    - final_result["waist_position"]
                )
                / final_result["rayleigh_range"]
            )
            ** 2
        )
        beam_radius_table = pd.DataFrame(
            {
                "Position from last lens [mm]": output_distances,
                "Beam radius [mm]": output_radii,
            }
        )

        plot_col, radius_table_col = st.columns([3, 2], vertical_alignment="top")
        with plot_col:
            st.pyplot(fig, width="content")
        with radius_table_col:
            st.markdown("**Final beam profile values**")
            st.dataframe(
                beam_radius_table,
                column_config={
                    "Position from last lens [mm]": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Beam radius [mm]": st.column_config.NumberColumn(format="%.6f"),
                },
                hide_index=True,
                height=280,
                width="stretch",
            )
            custom_output_distance = st.number_input(
                "Custom position from last lens [mm]",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.3f",
                key="custom_output_distance_mm",
            )
            custom_output_radius = final_result["waist_radius"] * np.sqrt(
                1
                + (
                    (
                        lens_positions[-1]
                        + custom_output_distance
                        - final_result["waist_position"]
                    )
                    / final_result["rayleigh_range"]
                )
                ** 2
            )
            st.session_state["custom_output_radius_mm"] = float(custom_output_radius)
            st.number_input(
                "Corresponding beam radius [mm]",
                format="%.6f",
                disabled=True,
                key="custom_output_radius_mm",
            )

with scan_tab:
    st.subheader("Single lens scan")
    st.caption(
        "Set a fixed target plane, then scan the position and focal length of one "
        "positive lens. Positions are measured from the incident beam waist at z = 0."
    )

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

    target_value_col, target_unit_col = st.columns([2, 1])
    scan_target_unit = target_unit_col.selectbox(
        "Target-distance unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_target_unit",
        on_change=convert_length_input_unit,
        args=("scan_target_input", "scan_target_unit"),
    )
    scan_target_input = target_value_col.number_input(
        f"Target position from incident waist [{scan_target_unit}]",
        min_value=1e-6 / LENGTH_TO_MM[scan_target_unit],
        value=1000.0 / LENGTH_TO_MM[scan_target_unit],
        step=10.0 / LENGTH_TO_MM[scan_target_unit],
        key="scan_target_input",
    )

    st.markdown("**Lens-position scan range**")
    min_value_col, min_unit_col, max_value_col, max_unit_col = st.columns([2, 1, 2, 1])
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
        min_value=0.0,
        value=50.0 / LENGTH_TO_MM[scan_min_unit],
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
        min_value=0.0,
        value=900.0 / LENGTH_TO_MM[scan_max_unit],
        step=10.0 / LENGTH_TO_MM[scan_max_unit],
        key="scan_max_input",
    )

    st.markdown("**Focal-length scan range**")
    f_min_value_col, f_min_unit_col, f_max_value_col, f_max_unit_col = st.columns(
        [2, 1, 2, 1]
    )
    scan_f_min_unit = f_min_unit_col.selectbox(
        "Minimum-focal-length unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_f_min_unit",
        on_change=convert_length_input_unit,
        args=("scan_f_min_input", "scan_f_min_unit"),
    )
    scan_f_min_input = f_min_value_col.number_input(
        f"Minimum focal length [{scan_f_min_unit}]",
        min_value=1e-6 / LENGTH_TO_MM[scan_f_min_unit],
        value=25.0 / LENGTH_TO_MM[scan_f_min_unit],
        step=1.0 / LENGTH_TO_MM[scan_f_min_unit],
        key="scan_f_min_input",
    )
    scan_f_max_unit = f_max_unit_col.selectbox(
        "Maximum-focal-length unit",
        GEOMETRY_UNITS,
        index=1,
        key="scan_f_max_unit",
        on_change=convert_length_input_unit,
        args=("scan_f_max_input", "scan_f_max_unit"),
    )
    scan_f_max_input = f_max_value_col.number_input(
        f"Maximum focal length [{scan_f_max_unit}]",
        min_value=1e-6 / LENGTH_TO_MM[scan_f_max_unit],
        value=1000.0 / LENGTH_TO_MM[scan_f_max_unit],
        step=10.0 / LENGTH_TO_MM[scan_f_max_unit],
        key="scan_f_max_input",
    )

    scan_lambda_mm = scan_lambda_input * LENGTH_TO_MM[scan_lambda_unit]
    scan_w0_mm = scan_w0_input * LENGTH_TO_MM[scan_w0_unit]
    scan_target_mm = scan_target_input * LENGTH_TO_MM[scan_target_unit]
    scan_min_mm = scan_min_input * LENGTH_TO_MM[scan_min_unit]
    scan_max_mm = scan_max_input * LENGTH_TO_MM[scan_max_unit]
    scan_f_min_mm = scan_f_min_input * LENGTH_TO_MM[scan_f_min_unit]
    scan_f_max_mm = scan_f_max_input * LENGTH_TO_MM[scan_f_max_unit]

    if scan_max_mm <= scan_min_mm:
        st.error("Maximum lens position must exceed minimum lens position.")
    elif scan_f_max_mm <= scan_f_min_mm:
        st.error("Maximum focal length must exceed minimum focal length.")
    elif scan_max_mm >= scan_target_mm:
        st.error("The complete lens-position scan range must be before the target plane.")
    else:
        scan = single_lens_scan(
            scan_lambda_mm,
            scan_w0_mm,
            scan_m2,
            scan_target_mm,
            scan_min_mm,
            scan_max_mm,
            scan_f_min_mm,
            scan_f_max_mm,
        )
        target_radius = scan["target_radius"]
        optimum_index = np.unravel_index(np.argmin(target_radius), target_radius.shape)

        result_cols = st.columns(4)
        with result_cols[0]:
            metric("Input Rayleigh range", scan["input_rayleigh"], "mm")
        with result_cols[1]:
            metric("Grid minimum target radius", target_radius[optimum_index], "mm")
        with result_cols[2]:
            metric("Lens position at minimum", scan["lens_grid"][optimum_index], "mm")
        with result_cols[3]:
            metric("Focal length at minimum", scan["focal_grid"][optimum_index], "mm")

        scale_col, explanation_col = st.columns([1, 3], vertical_alignment="bottom")
        color_scale = scale_col.selectbox(
            "Beam-radius color scale",
            ["Logarithmic", "Linear"],
            key="scan_color_scale",
        )
        explanation_col.caption(
            "Contours show the absolute target-to-waist separation divided by the "
            "transformed Rayleigh range."
        )

        slice_control_col, slice_value_col = st.columns([1, 2], vertical_alignment="bottom")
        slice_dimension = slice_control_col.segmented_control(
            "Slice at fixed",
            ["Lens position", "Focal length"],
            default="Lens position",
            required=True,
            key="scan_slice_dimension",
        )
        if slice_dimension == "Lens position":
            slice_value_key = "scan_slice_lens_position_mm"
            slice_value_default = (scan_min_mm + scan_max_mm) / 2
            if slice_value_key not in st.session_state or not (
                scan_min_mm
                <= st.session_state[slice_value_key]
                <= scan_max_mm
            ):
                st.session_state[slice_value_key] = slice_value_default
            fixed_slice_value = slice_value_col.number_input(
                "Fixed lens position [mm]",
                min_value=float(scan_min_mm),
                max_value=float(scan_max_mm),
                step=float((scan_max_mm - scan_min_mm) / 240),
                key=slice_value_key,
            )
            slice_coordinates = scan["focal_lengths"]
            q_at_slice_lens = fixed_slice_value + 1j * scan["input_rayleigh"]
            q_after_slice_lens = 1 / (
                1 / q_at_slice_lens - 1 / slice_coordinates
            )
            q_at_slice_target = (
                q_after_slice_lens + scan_target_mm - fixed_slice_value
            )
            slice_xlabel = "Focal length [mm]"
            slice_title = f"Slice at lens position = {fixed_slice_value:.6g} mm"
        else:
            slice_value_key = "scan_slice_focal_length_mm"
            slice_value_default = (scan_f_min_mm + scan_f_max_mm) / 2
            if slice_value_key not in st.session_state or not (
                scan_f_min_mm
                <= st.session_state[slice_value_key]
                <= scan_f_max_mm
            ):
                st.session_state[slice_value_key] = slice_value_default
            fixed_slice_value = slice_value_col.number_input(
                "Fixed focal length [mm]",
                min_value=float(scan_f_min_mm),
                max_value=float(scan_f_max_mm),
                step=float((scan_f_max_mm - scan_f_min_mm) / 240),
                key=slice_value_key,
            )
            slice_coordinates = scan["lens_positions"]
            q_at_slice_lens = slice_coordinates + 1j * scan["input_rayleigh"]
            q_after_slice_lens = 1 / (
                1 / q_at_slice_lens - 1 / fixed_slice_value
            )
            q_at_slice_target = (
                q_after_slice_lens + scan_target_mm - slice_coordinates
            )
            slice_xlabel = "Lens position from incident waist [mm]"
            slice_title = f"Slice at focal length = {fixed_slice_value:.6g} mm"

        slice_target_radius = np.sqrt(
            scan_m2
            * scan_lambda_mm
            / np.pi
            * (q_at_slice_target.real**2 + q_at_slice_target.imag**2)
            / q_at_slice_target.imag
        )
        slice_minimum_index = int(np.argmin(slice_target_radius))

        fig, axis = plt.subplots(figsize=(5.5, 3.2))
        color_norm = (
            LogNorm(vmin=target_radius.min(), vmax=target_radius.max())
            if color_scale == "Logarithmic"
            else None
        )
        radius_map = axis.pcolormesh(
            scan["lens_positions"],
            scan["focal_lengths"],
            target_radius,
            shading="auto",
            cmap="viridis",
            norm=color_norm,
        )
        offset_ratio = scan["target_offset_in_rayleigh"]
        contour_specs = [
            (0.5, "white", r"$|\Delta z|=z_R'/2$"),
            (1.0, "#ff7f0e", r"$|\Delta z|=z_R'$"),
        ]
        legend_handles = []
        missing_contours = []
        for level, color, label in contour_specs:
            if offset_ratio.min() <= level <= offset_ratio.max():
                contour = axis.contour(
                    scan["lens_grid"],
                    scan["focal_grid"],
                    offset_ratio,
                    levels=[level],
                    colors=[color],
                    linewidths=1.1,
                )
                axis.clabel(contour, fmt={level: label}, fontsize=7)
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=1.2, label=label)
                )
            else:
                missing_contours.append(label)
        axis.scatter(
            scan["lens_grid"][optimum_index],
            scan["focal_grid"][optimum_index],
            marker="x",
            s=30,
            color="red",
            linewidth=1.2,
            label="Minimum target radius",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="red",
                marker="x",
                linestyle="None",
                label="Grid minimum radius",
            )
        )
        if slice_dimension == "Lens position":
            axis.axvline(
                fixed_slice_value,
                color="#e7298a",
                linestyle="--",
                linewidth=1.2,
            )
        else:
            axis.axhline(
                fixed_slice_value,
                color="#e7298a",
                linestyle="--",
                linewidth=1.2,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#e7298a",
                linestyle="--",
                lw=1.2,
                label="Selected slice",
            )
        )
        axis.set(
            title=f"Beam radius at target z = {scan_target_mm:.6g} mm",
            xlabel="Lens position from incident waist [mm]",
            ylabel="Focal length [mm]",
        )
        axis.legend(
            handles=legend_handles,
            #facecolor="#333333",
            #labelcolor="white",
        )
        colorbar = fig.colorbar(radius_map, ax=axis, pad=0.02)
        colorbar.set_label("Beam radius at target [mm]")
        fig.tight_layout()

        slice_fig, slice_axis = plt.subplots(figsize=(5.5, 3.2))
        slice_axis.plot(slice_coordinates, slice_target_radius, color="#e7298a")
        slice_axis.scatter(
            slice_coordinates[slice_minimum_index],
            slice_target_radius[slice_minimum_index],
            marker="x",
            s=28,
            color="red",
            linewidth=1.2,
            zorder=3,
        )
        slice_axis.set(
            title=slice_title,
            xlabel=slice_xlabel,
            ylabel="Beam radius at target [mm]",
        )
        slice_axis.grid(alpha=0.3)
        slice_fig.tight_layout()

        map_col, slice_plot_col = st.columns(2, vertical_alignment="top")
        with map_col:
            st.pyplot(fig, width="content")
            if missing_contours:
                st.info(
                    "The following contour does not occur inside the selected scan "
                    "range: "
                    + ", ".join(missing_contours)
                    + "."
                )
        with slice_plot_col:
            st.pyplot(slice_fig, width="content")
            st.caption(
                f"Slice minimum: {slice_target_radius[slice_minimum_index]:.6g} mm "
                f"at {slice_xlabel.removesuffix(' [mm]').lower()} = "
                f"{slice_coordinates[slice_minimum_index]:.6g} mm."
            )

        plt.close(fig)
        plt.close(slice_fig)

        scan_csv = pd.DataFrame({
            "lens_position_mm": scan["lens_grid"].ravel(),
            "focal_length_mm": scan["focal_grid"].ravel(),
            "target_position_mm": scan_target_mm,
            "target_beam_radius_mm": target_radius.ravel(),
            "output_waist_position_mm": scan["output_waist_position"].ravel(),
            "output_rayleigh_range_mm": scan["output_rayleigh"].ravel(),
            "output_waist_radius_mm": scan["output_waist_radius"].ravel(),
            "absolute_target_offset_in_rayleigh_ranges": scan[
                "target_offset_in_rayleigh"
            ].ravel(),
        }).to_csv(index=False)
        st.download_button(
            "Download scan data",
            scan_csv,
            "single_lens_scan.csv",
            "text/csv",
        )

st.divider()
st.caption("Model: paraxial thin lens, stigmatic propagation, and constant M². Confirm unit conventions before using results in an experiment.")
