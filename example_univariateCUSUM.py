from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor, JumpbackWindow, ResetWindow
from streamchange.utils.example_data import three_segments_data

seg_len = 100000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=10)[0]

test = UnivariateCUSUM(minsl=1).set_default_threshold(10 * df.size)
window = JumpbackWindow(4, 100)
detector = WindowSegmentor(test, window)
cpts = []
for t, x in df.items():
    detector.update({df.name: x})
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)