"""clock - the computer's current local date and time (models have no clock of their own)."""
import time

NAME = "clock"
DESCRIPTION = "Return the current local date, time, weekday and UTC offset of this computer."
PARAMETERS = {"type": "object", "properties": {}, "required": []}


def run() -> str:
    return time.strftime("%A %Y-%m-%d %H:%M:%S (UTC%z)")
