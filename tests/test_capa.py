from river.stream import iter_pandas

from streamchange.data import simulate
from streamchange.capa import Capa, ConstMeanL2


def test_accuracy():
    seg_len = 100
    df = simulate([0, 30], [seg_len], p=1, seed=2)
    point_anom_pos = 10
    df.iloc[point_anom_pos] = 200
    capa = Capa(ConstMeanL2(), minsl=2, maxsl=1000)
    collective_anoms, point_anoms = capa.fit(df).predict()
    assert len(collective_anoms) == 1
    assert collective_anoms[0]["end"] == df.size - 1
    assert collective_anoms[0]["start"] == seg_len
    assert len(point_anoms) == 1
    assert point_anoms[0]["start"] == point_anom_pos


# def test_capa_penalty():
#     seg_len = 100
#     df = simulate([0, 10, 0], [seg_len], p=1, seed=5)
#     thresholds = [0.001, 0.1, 1, 2, 3, 4, 5, 10, 10000]
#     for threshold in thresholds:
#         try:
#             test = UnivariateCUSUM(minsl=1, threshold=threshold)
#             detector = WindowSegmentor(test, 2, 100)
#             for t, (x, _) in enumerate(iter_pandas(df)):
#                 detector.update(x)
#         except Exception as exc:
#             assert False, f"'detector.update()' raised exception {exc}"


# def test_capa_seglens():
#     seg_len = 20
#     df = simulate([0, 10, 0, 20, 0, 1, 0, 3, 5, 0, 4], [seg_len], p=1, seed=34)
#     test = UnivariateCUSUM(minsl=1, threshold=10)
#     detector = WindowSegmentor(test, 4, 20)
#     most_recent_cpt = -1
#     for t, (x, _) in enumerate(iter_pandas(df)):
#         detector.update(x)
#         assert len(detector.window) <= detector.max_window
#         assert len(detector.window) <= -most_recent_cpt
#         if detector.change_detected:
#             most_recent_cpt = detector.changepoints[-1]
#         else:
#             most_recent_cpt -= 1
