//! Plotting helpers for the KdV PINN sample, using the `plotters` crate.
//!
//! The plots draw no text: `plotters` is built without a font backend, because
//! its `ttf` backend needs the FreeType and fontconfig system libraries on
//! Linux, and drawing a caption, axis label or legend without one panics
//! (#1968). Grid lines mark the axes; the tutorial states the axis ranges and
//! the line colours.

use plotters::prelude::*;
use std::error::Error;

#[cfg(test)]
mod tests;

/// Compute `(lower, upper)` y-axis bounds for a log-scale loss plot.
///
/// A logarithmic axis can only display strictly positive, finite values, so
/// non-positive and non-finite entries are ignored when scanning for the
/// minimum and maximum. A multiplicative margin keeps the smallest and largest
/// losses off the chart border. If `history` contains no positive finite value
/// (e.g. it is empty), a safe default decade `(1e-8, 1.0)` is returned.
pub(crate) fn loss_axis_bounds(history: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in history {
        if v.is_finite() && v > 0.0 {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (1e-8, 1.0);
    }
    (lo / 1.5, hi * 1.5)
}

/// Write the training-loss curve to a PNG file with a logarithmic y-axis.
///
/// `history` holds the loss value recorded at each epoch. The y-axis uses a log
/// scale because the loss spans several orders of magnitude over training; any
/// non-positive or non-finite value is clamped to the lower bound so it can be
/// drawn. The image is 800×600 pixels; the red curve is the loss, and the
/// horizontal grid lines sit on the log-scale y-axis.
pub(crate) fn write_loss_png(path: &str, history: &[f64]) -> Result<(), Box<dyn Error>> {
    const W: u32 = 800;
    const H: u32 = 600;

    let root = BitMapBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE)?;

    let (y_lo, y_hi) = loss_axis_bounds(history);
    let x_hi = history.len().max(1) as f64;

    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .build_cartesian_2d(0.0..x_hi, (y_lo..y_hi).log_scale())?;

    chart.configure_mesh().disable_axes().draw()?;

    chart.draw_series(LineSeries::new(
        history.iter().enumerate().map(|(i, &l)| {
            let y = if l.is_finite() && l > 0.0 { l } else { y_lo };
            (i as f64, y)
        }),
        RED.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Write an animated GIF comparing the analytic and predicted solution curves.
///
/// `xs` is the spatial grid. `frames` is a slice of `(t, analytic, predicted)`
/// values at each animation frame. The GIF uses 640×480 pixels with a delay of
/// 100 ms per frame. Each frame spans `x` in `[-5, 5]` and `u` in `[-0.5, 2.5]`;
/// the blue curve is the analytic solution and the red curve the prediction.
pub(crate) fn write_comparison_gif(
    path: &str,
    xs: &[f64],
    frames: &[(f64, Vec<f64>, Vec<f64>)],
) -> Result<(), Box<dyn Error>> {
    const W: u32 = 640;
    const H: u32 = 480;

    let backend = BitMapBackend::gif(path, (W, H), 100)?;
    let root = backend.into_drawing_area();

    for (_t, analytic, predicted) in frames {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .build_cartesian_2d(-5.0..5.0, -0.5..2.5)?;

        chart
            .configure_mesh()
            .x_labels(11)
            .y_labels(7)
            .disable_axes()
            .draw()?;

        chart.draw_series(LineSeries::new(
            xs.iter().zip(analytic.iter()).map(|(&x, &u)| (x, u)),
            BLUE.stroke_width(2),
        ))?;

        chart.draw_series(LineSeries::new(
            xs.iter().zip(predicted.iter()).map(|(&x, &u)| (x, u)),
            RED.stroke_width(2),
        ))?;

        root.present()?;
    }

    Ok(())
}
