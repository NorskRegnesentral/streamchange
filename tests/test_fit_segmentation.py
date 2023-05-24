def test_fit_segmentation():
    from river.stats import Mean, Quantile
    import pandas as pd
    import numpy as np

    from streamchange.amoc import WindowSegmentor, CUSUM

    # from streamchange.sequential import LordenPollakCUSUM
    from streamchange.segment_stats import StatUnion, StatBuffer
    from streamchange.conveniences import fit_segmentation
    from streamchange.data import simulate

    seg_len = 50
    series = simulate([0, 10, 0], [seg_len], p=1)[0]
    test = CUSUM(0)
    detector = WindowSegmentor(test, 2, 100)
    stat = StatUnion({"mean": StatBuffer(Mean())}, detector.max_window)
    segmentation = fit_segmentation(detector, stat, series)
    assert len(segmentation) == series.size

    test = CUSUM(20)
    detector = WindowSegmentor(test, 5, 100)
    segmentation = fit_segmentation(detector, stat.reset(), series)
    assert len(segmentation) == 3
