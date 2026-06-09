#!/usr/bin/env python3

from src.control.state_machine import DrivingState, DrivingStateMachine


def test_turn_done_clears_pending_sign_and_target():
    sm = DrivingStateMachine({"turn_duration": 0.0})
    sm.state = DrivingState.TURN
    sm.pending_sign = "right"
    sm.approach_target_x = 700
    sm.approach_target_y = 300
    sm._boundary_hit = False

    out = sm.update()

    assert out["state"] == "CRUISE"
    assert out["pending_sign"] is None
    assert out["approach_target_x"] is None
    assert out["approach_target_y"] is None


def test_approach_updates_to_closer_high_confidence_sign():
    sm = DrivingStateMachine()
    sm.state = DrivingState.APPROACH
    sm.pending_sign = "left"
    sm.approach_target_x = 768
    sm.approach_target_y = 387

    out = sm.update(sign_result={
        "signs": [{
            "type": "right",
            "x": 750,
            "y": 556,
            "score": 0.934,
        }]
    })

    assert out["state"] == "APPROACH"
    assert out["pending_sign"] == "right"
    assert out["approach_target_x"] == 750
    assert out["approach_target_y"] == 556
