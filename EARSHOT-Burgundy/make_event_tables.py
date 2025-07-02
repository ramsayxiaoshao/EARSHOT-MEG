# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# Generate subject-specific event tables needed for creating predictors
from pathlib import Path

from eelbrain import *
import burgundy


EVENT_DIR = Path('~/Data/Burgundy/events').expanduser()
EVENT_DIR.mkdir(exist_ok=True)

for subject in burgundy.e:
    dst = EVENT_DIR / f"{subject}.pickle"
    events = burgundy.e.load_selected_events()
    save.pickle(events, dst)
