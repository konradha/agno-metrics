import numpy as np
from types import SimpleNamespace
from cli import cmd_calibrate, cmd_evaluate, cmd_curate


def test_cli_calibrate_and_evaluate(tmp_path):
    pred = np.linspace(0, 1, 100)
    tgt = pred**0.8
    pred_path = tmp_path / "pred.npy"
    tgt_path = tmp_path / "tgt.npy"
    np.save(pred_path, pred)
    np.save(tgt_path, tgt)
    cal_out = tmp_path / "artifacts" / "cal.pkl"
    met_out = tmp_path / "artifacts" / "met.json"
    plot_out = tmp_path / "artifacts" / "cal.png"
    args = SimpleNamespace(
        pred_path=str(pred_path),
        tgt_path=str(tgt_path),
        out_calibrator=str(cal_out),
        out_metrics=str(met_out),
        out_plot=str(plot_out),
    )
    cmd_calibrate(args)
    assert cal_out.exists() and met_out.exists() and plot_out.exists()

    met2_out = tmp_path / "artifacts" / "met2.json"
    plot2_out = tmp_path / "artifacts" / "scatter.png"
    args2 = SimpleNamespace(
        pred_path=str(pred_path),
        tgt_path=str(tgt_path),
        out_metrics=str(met2_out),
        out_plot=str(plot2_out),
    )
    cmd_evaluate(args2)
    assert met2_out.exists() and plot2_out.exists()


def test_cli_curate(tmp_path):
    emb = np.random.RandomState(0).randn(20, 8).astype(np.float32)
    emb_path = tmp_path / "emb.npy"
    np.save(emb_path, emb)
    sel_out = tmp_path / "sel.npy"
    args = SimpleNamespace(emb_path=str(emb_path), keep=5, selection_out=str(sel_out))
    cmd_curate(args)
    assert sel_out.exists()
    sel = np.load(sel_out)
    assert sel.shape == (5,)
