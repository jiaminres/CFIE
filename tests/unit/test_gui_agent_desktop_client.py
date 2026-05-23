from __future__ import annotations

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_DO_NOT_REPLY,
    DesktopClientState,
    MacroConfig,
    TargetAppConfig,
    parse_macro_sequence,
)


def test_parse_macro_sequence_groups_combo_keys():
    assert parse_macro_sequence("CTRL+A, B, SHIFT + C") == (
        ("CTRL", "A"),
        ("B",),
        ("SHIFT", "C"),
    )


def test_desktop_state_starts_without_seed_data():
    state = DesktopClientState()

    assert state.target_apps == {}
    assert state.job_board.jobs == {}
    assert state.human_loop.list_requests(include_completed=True) == ()
    assert state.action_macros.macros == {}


def test_desktop_state_registers_macro_for_context_payload():
    state = DesktopClientState()
    state.register_macro(
        MacroConfig(
            name="quick_reply",
            description="Select all text and type a shortcut marker.",
            sequence="CTRL+A, B",
            scope="app",
            app_id="app_shop",
        )
    )

    payload = state.action_macros.to_context_payload()

    assert payload["macros"][0]["name"] == "quick_reply"
    assert payload["macros"][0]["metadata"]["sequence"] == "CTRL+A, B"
    assert payload["macros"][0]["metadata"]["scope"] == "app"
    assert payload["macros"][0]["metadata"]["app_id"] == "app_shop"


def test_structured_human_reply_preserves_direct_command():
    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(app_id="app_shop", app_name="Shop", job_id="job_shop")
    )
    request = state.human_loop.request_help(
        question="Should I reply?",
        task_id="seller_question",
        metadata={"job_id": "job_shop"},
    )

    task = state.submit_structured_human_reply(
        request_id=request.request_id,
        manager_input="Do not answer this seller. Move to the next conversation.",
        decision_type="更改路径",
        direct_command=DIRECT_COMMAND_DO_NOT_REPLY,
        constraints="Do not mention AI.",
    )

    reply = task["reply"]
    structured = reply["metadata"]["structured_payload"]
    assert "不要回复当前对象" in reply["text"]
    assert structured["direct_command"] == DIRECT_COMMAND_DO_NOT_REPLY
    assert structured["decision_type"] == "更改路径"
    assert structured["constraints"] == ["Do not mention AI."]
