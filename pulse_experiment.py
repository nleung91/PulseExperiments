from slab import InstrumentManager


class Experiment:
    def __init__(self, cfg):
        im = InstrumentManager()

        self.tek = im['TEK']


    def run_experiment(self, sequences):
        print(self.tek.get_id())