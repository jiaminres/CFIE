from __future__ import annotations

import ctypes

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

_KEYS = {
    "ALT": 0x12,
    "ARROWDOWN": 0x28,
    "ARROWLEFT": 0x25,
    "ARROWRIGHT": 0x27,
    "ARROWUP": 0x26,
    "BACKSPACE": 0x08,
    "CTRL": 0x11,
    "CONTROL": 0x11,
    "DELETE": 0x2E,
    "END": 0x23,
    "ENTER": 0x0D,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "HOME": 0x24,
    "META": 0x5B,
    "PAGEDOWN": 0x22,
    "PAGEUP": 0x21,
    "SHIFT": 0x10,
    "SPACE": 0x20,
    "TAB": 0x09,
    "WIN": 0x5B,
}

for _index in range(1, 13):
    _KEYS[f"F{_index}"] = 0x6F + _index


ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUTUNION),
    ]


def _vk_for_key(key: str) -> int:
    normalized = key.replace("_", "").replace("-", "").upper()
    if normalized in _KEYS:
        return _KEYS[normalized]
    if len(key) == 1:
        vk = ctypes.windll.user32.VkKeyScanW(ord(key)) & 0xFF
        if vk:
            return int(vk)
    raise ValueError(f"Unsupported key: {key}")


def key_down(key: str) -> None:
    ctypes.windll.user32.keybd_event(_vk_for_key(key), 0, 0, 0)


def key_up(key: str) -> None:
    ctypes.windll.user32.keybd_event(_vk_for_key(key), 0, KEYEVENTF_KEYUP, 0)


def press_keys(keys: tuple[str, ...]) -> None:
    if len(keys) > 1 and any(key.upper() in {"CTRL", "CONTROL", "ALT", "SHIFT", "META", "WIN"} for key in keys):
        pressed: list[str] = []
        try:
            for key in keys:
                key_down(key)
                pressed.append(key)
        finally:
            for key in reversed(pressed):
                key_up(key)
        return

    for key in keys:
        key_down(key)
        key_up(key)


def type_text(text: str) -> None:
    for char in text:
        code = ord(char)
        inputs = (INPUT * 2)(
            INPUT(
                type=INPUT_KEYBOARD,
                union=INPUTUNION(
                    ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0),
                ),
            ),
            INPUT(
                type=INPUT_KEYBOARD,
                union=INPUTUNION(
                    ki=KEYBDINPUT(
                        0,
                        code,
                        KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                        0,
                        0,
                    ),
                ),
            ),
        )
        ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
