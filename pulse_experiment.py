from slab import InstrumentManager

def run_experiment():
    im = InstrumentManager()

    tek = im['TEK']

    print(tek.get_id())