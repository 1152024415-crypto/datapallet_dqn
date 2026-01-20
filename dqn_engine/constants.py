from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# Constants
# ============================================================================

LOCATIONS = ["unknown", "other", "Work", "Subway Station", "Park", "Street"]

SCENES = [
    "unknown",
    "other",
    # office:
    "office_cubicles",
    "office",
    "computer_room",
    "home_office",
    # meeting:
    "conference_room",
    "conference_center",
    "lecture_room",
    # Dinning:
    "dining_room",
    "kitchen",
    "cafeteria",
    "food_court",
    "pantry",
    "dining_hall",
    "booth_indoor",
    # Walking outdoor:
    "park",
    "street",
    "crosswalk",
    "promenade",
    # Subway platform:
    "subway_platform",
]

ACTIVITIES = ["unknown", "stationary", "slow walk", "fast walk", "fast run", "elevator"]
SOUND_LEVELS = ["unknown", "very_quiet", "soft", "normal", "noisy", "very_noisy"]
LIGHT_LEVELS = ["unknown", "extremely_dark", "dim", "moderate", "bright", "harsh"]

# Probe Actions and Costs
PROBE_ACTIONS = ["QUERY_LOC_GPS", "QUERY_VISUAL"]
PROBE_COST = {"QUERY_LOC_GPS": 0.25, "QUERY_VISUAL": 1.00}

# Recommend Actions
RECOMMEND_ACTIONS = [
    "NONE",
    "step_count",
    "relax",
    "transit_QR_code",
    "silent_DND",
    "Play Music/news"
]

# Scenario Priority Order (higher priority = checked first)
PRIORITY = [
    "TRANSIT_QR_CODE",
    "SILENT_DND",
    "STEP_COUNT",
    "RELAX",
    "PLAY_MUSIC/NEWS"
]

# Action Mappings
ALL_ACTIONS = PROBE_ACTIONS + RECOMMEND_ACTIONS
ACTION_TO_ID = {a: i for i, a in enumerate(ALL_ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Gate:
    """Gate state for scenario activation tracking."""
    active: bool = False
    fired: bool = False
    off_counter: int = 999


@dataclass
class TruthStep:
    """Ground truth state at a single time step."""
    t: int
    # Activity: [(act, act_dur), (act, act_dur), ...]
    activities: List[Tuple[int, int]]
    # Location: [(loc, loc_dur), (loc, loc_dur), ...]
    locations: List[Tuple[int, int]]
    # Scene: [(scene, scene_dur), (scene, scene_dur), ...]
    scenes: List[Tuple[int, int]]
    # Light: [(light, light_dur), (light, light_dur), ...]
    lights: List[Tuple[int, int]]


# ============================================================================
# Default User Profiles
# ============================================================================

DEFAULT_PROFILES = [
]


# ============================================================================
# Utility Functions
# ============================================================================

def _clip_int(x: int, lo: int, hi: int) -> int:
    """Clip integer value to range [lo, hi]."""
    return max(lo, min(hi, int(x)))


def _fmt_hhmm(t_min: int) -> str:
    """Format minutes since midnight as HH:MM."""
    h, m = divmod(int(t_min), 60)
    return f"{h:02d}:{m:02d}"


TRAIN_EPISODES = 8000
EVAL_EPISODES = 2000