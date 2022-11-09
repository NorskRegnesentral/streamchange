import cProfile, pstats, io
from pstats import SortKey


class Profiler:
    def __init__(self):
        self.pr = cProfile.Profile()
        pass

    def start(self):
        self.pr.enable()

    def stop(self):
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
