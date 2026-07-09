"""
Patch NeuroCogEEG non-TMT task pipelines so response-locked epochs are
created only from valid immediately preceding stimulus -> response pairs.

This script modifies:
    pipelines/flanker.py
    pipelines/gonogo.py
    pipelines/readysetgo.py

It creates one backup per file the first time it runs:
    <file>.pre_response_context_filter.bak

Run from the project root:
    python pipelines/apply_response_locked_context_filter.py
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HELPER_CODE = r'''

def filter_events_by_immediate_preceding_context(
    events: np.ndarray,
    valid_preceding_codes_by_response_code: dict[int, tuple[int, ...]],
) -> np.ndarray:
    """
    Keep response events only when they are immediately preceded by a valid
    task stimulus event.

    This prevents stray or repeated response markers from entering
    response-locked ERN/RP/PMP analyses. Raw EDF marker channels are not
    modified; filtering is applied only at the derived epoch-selection layer.
    """
    selected_events = []

    for index in range(1, len(events)):
        current_event = events[index]
        current_code = int(current_event[2])

        if current_code not in valid_preceding_codes_by_response_code:
            continue

        previous_code = int(events[index - 1][2])
        valid_previous_codes = valid_preceding_codes_by_response_code[
            current_code
        ]

        if previous_code in valid_previous_codes:
            selected_events.append(current_event)

    if not selected_events:
        return np.empty((0, 3), dtype=int)

    return np.asarray(selected_events, dtype=int)
'''

FLANKER_CREATE_RESPONSE = r'''
def create_response_locked_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create response-locked epochs for valid Flanker stimulus-response pairs.

    Only response markers immediately preceded by a Flanker stimulus marker
    are used. This excludes stray or repeated response markers from ERN/RP/PMP
    analyses without modifying the raw EDF marker channel.
    """
    events_config = experiment_config["events"]

    stimulus_code = get_event_code(
        events_config=events_config,
        event_name="stimulus",
    )
    correct_code = get_event_code(
        events_config=events_config,
        event_name="correct_response",
    )
    error_code = get_event_code(
        events_config=events_config,
        event_name="error_response",
    )

    valid_response_events = filter_events_by_immediate_preceding_context(
        events=events,
        valid_preceding_codes_by_response_code={
            correct_code: (stimulus_code,),
            error_code: (stimulus_code,),
        },
    )

    epoch_config = experiment_config["epochs"]["response_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=valid_response_events,
        event_id={
            "correct_response": correct_code,
            "error_response": error_code,
        },
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )
'''

GONOGO_CREATE_RESPONSE = r'''
def create_response_locked_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create response-locked epochs for valid Go/No-Go stimulus-response pairs.

    Correct go responses are included only when immediately preceded by a
    go_stimulus marker. False alarms are included only when immediately
    preceded by a nogo_stimulus marker. This excludes stray response markers
    from response-locked RP/PMP analyses without modifying raw EDF data.
    """
    events_config = experiment_config["events"]

    go_stimulus_code = get_event_code(
        events_config=events_config,
        event_name="go_stimulus",
    )
    nogo_stimulus_code = get_event_code(
        events_config=events_config,
        event_name="nogo_stimulus",
    )
    correct_code = get_event_code(
        events_config=events_config,
        event_name="correct_response",
    )
    false_alarm_code = get_event_code(
        events_config=events_config,
        event_name="false_alarm",
    )

    valid_response_events = filter_events_by_immediate_preceding_context(
        events=events,
        valid_preceding_codes_by_response_code={
            correct_code: (go_stimulus_code,),
            false_alarm_code: (nogo_stimulus_code,),
        },
    )

    epoch_config = experiment_config["epochs"]["response_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=valid_response_events,
        event_id={
            "correct_response": correct_code,
            "error_response": false_alarm_code,
        },
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )
'''

READYSETGO_CREATE_RESPONSE = r'''
def create_response_locked_epochs(
    raw_clean,
    events: np.ndarray,
    experiment_config: dict[str, Any],
):
    """
    Create response-locked epochs for valid ReadySetGo go-response pairs.

    Go-response markers are included only when immediately preceded by the
    go marker. This excludes stray response markers from response-locked
    RP/PMP analyses without modifying raw EDF data.
    """
    events_config = experiment_config["events"]

    go_code = get_event_code(
        events_config=events_config,
        event_name="go",
    )
    response_code = get_event_code(
        events_config=events_config,
        event_name="go_response",
    )

    valid_response_events = filter_events_by_immediate_preceding_context(
        events=events,
        valid_preceding_codes_by_response_code={
            response_code: (go_code,),
        },
    )

    epoch_config = experiment_config["epochs"]["response_locked"]
    reject_criteria = get_reject_criteria(
        experiment_config["quality_control"]
    )

    return create_epochs(
        raw=raw_clean,
        events=valid_response_events,
        event_id={"go_response": response_code},
        epoch_config=epoch_config,
        reject_criteria=reject_criteria,
        picks="eeg",
    )
'''

PATCHES = {
    PROJECT_ROOT / "pipelines" / "flanker.py": FLANKER_CREATE_RESPONSE,
    PROJECT_ROOT / "pipelines" / "gonogo.py": GONOGO_CREATE_RESPONSE,
    PROJECT_ROOT / "pipelines" / "readysetgo.py": READYSETGO_CREATE_RESPONSE,
}


def insert_helper_if_needed(text: str) -> str:
    if "def filter_events_by_immediate_preceding_context" in text:
        return text

    marker = "\ndef create_response_locked_epochs("
    if marker not in text:
        raise RuntimeError(
            "Could not find create_response_locked_epochs insertion point."
        )

    return text.replace(marker, HELPER_CODE + marker, 1)


def replace_create_response_function(text: str, replacement: str) -> str:
    pattern = re.compile(
        r"\ndef create_response_locked_epochs\([\s\S]*?(?=\n\ndef compute_)",
        flags=re.MULTILINE,
    )

    new_text, count = pattern.subn("\n" + replacement.strip() + "\n", text, count=1)

    if count != 1:
        raise RuntimeError(
            f"Expected to replace exactly one create_response_locked_epochs "
            f"function, replaced {count}."
        )

    return new_text


def patch_file(path: Path, replacement: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    backup_path = path.with_suffix(path.suffix + ".pre_response_context_filter.bak")
    if not backup_path.exists():
        backup_path.write_text(text, encoding="utf-8")

    text = insert_helper_if_needed(text)
    text = replace_create_response_function(text, replacement)

    path.write_text(text, encoding="utf-8")
    print(f"Updated: {path.relative_to(PROJECT_ROOT)}")
    print(f"Backup : {backup_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    for path, replacement in PATCHES.items():
        patch_file(path, replacement)

    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    main()
