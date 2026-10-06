use super::*;

#[test]
fn loss_axis_bounds_brackets_positive_data() {
    let (lo, hi) = loss_axis_bounds(&[1.0, 0.5, 0.001]);
    assert!(lo > 0.0, "lower bound must stay positive for a log axis");
    assert!(lo <= 0.001, "lower bound must not exceed the smallest loss");
    assert!(hi >= 1.0, "upper bound must not be below the largest loss");
}

#[test]
fn loss_axis_bounds_ignores_nonpositive_and_nonfinite() {
    let (lo, hi) = loss_axis_bounds(&[f64::NAN, 0.0, -1.0, 0.01, f64::INFINITY]);
    assert!(lo > 0.0 && lo.is_finite());
    assert!(hi.is_finite());
    assert!(lo <= 0.01 && hi >= 0.01);
}

#[test]
fn loss_axis_bounds_defaults_when_no_positive_value() {
    assert_eq!(loss_axis_bounds(&[]), (1e-8, 1.0));
    assert_eq!(loss_axis_bounds(&[f64::NAN, 0.0, -2.0]), (1e-8, 1.0));
}

/// Both plots render without a font backend (#1968): drawing any text would
/// panic in `plotters` here.
#[test]
fn plots_render_without_a_font_backend() {
    let dir = std::env::temp_dir().join(format!("kdv-pinn-plot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("loss.png");
    let gif = dir.join("kdv.gif");

    write_loss_png(png.to_str().unwrap(), &[1.0, 0.1, 0.01, 0.001]).unwrap();
    let xs = vec![-1.0, 0.0, 1.0];
    let frames = vec![
        (0.0, vec![0.0, 1.0, 0.0], vec![0.1, 0.9, 0.1]),
        (0.5, vec![0.0, 0.5, 1.0], vec![0.1, 0.6, 0.9]),
    ];
    write_comparison_gif(gif.to_str().unwrap(), &xs, &frames).unwrap();

    assert!(std::fs::metadata(&png).unwrap().len() > 0);
    assert!(std::fs::metadata(&gif).unwrap().len() > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}
