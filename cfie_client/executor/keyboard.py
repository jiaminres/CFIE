from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
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


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUTUNION),
    ]


_USER32 = None
_KERNEL32 = None


def _user32():
    global _USER32
    if _USER32 is None:
        _USER32 = ctypes.WinDLL("user32", use_last_error=True)
        _USER32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        _USER32.SendInput.restype = wintypes.UINT
        _USER32.VkKeyScanW.argtypes = (wintypes.WCHAR,)
        _USER32.VkKeyScanW.restype = ctypes.c_short
        _USER32.keybd_event.argtypes = (
            wintypes.BYTE,
            wintypes.BYTE,
            wintypes.DWORD,
            ULONG_PTR,
        )
        _USER32.keybd_event.restype = None
        _USER32.OpenClipboard.argtypes = (wintypes.HWND,)
        _USER32.OpenClipboard.restype = wintypes.BOOL
        _USER32.CloseClipboard.argtypes = ()
        _USER32.CloseClipboard.restype = wintypes.BOOL
        _USER32.EmptyClipboard.argtypes = ()
        _USER32.EmptyClipboard.restype = wintypes.BOOL
        _USER32.IsClipboardFormatAvailable.argtypes = (wintypes.UINT,)
        _USER32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        _USER32.GetClipboardData.argtypes = (wintypes.UINT,)
        _USER32.GetClipboardData.restype = wintypes.HANDLE
        _USER32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
        _USER32.SetClipboardData.restype = wintypes.HANDLE
    return _USER32


def _kernel32():
    global _KERNEL32
    if _KERNEL32 is None:
        _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _KERNEL32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
        _KERNEL32.GlobalAlloc.restype = wintypes.HGLOBAL
        _KERNEL32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
        _KERNEL32.GlobalLock.restype = ctypes.c_void_p
        _KERNEL32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
        _KERNEL32.GlobalUnlock.restype = wintypes.BOOL
        _KERNEL32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
        _KERNEL32.GlobalFree.restype = wintypes.HGLOBAL
    return _KERNEL32


def _send_keyboard_inputs(inputs: tuple[KEYBDINPUT, ...]) -> None:
    array_type = INPUT * len(inputs)
    array = array_type(
        *[
            INPUT(
                type=INPUT_KEYBOARD,
                union=INPUTUNION(ki=input_),
            )
            for input_ in inputs
        ]
    )
    sent = _user32().SendInput(len(array), array, ctypes.sizeof(INPUT))
    if sent != len(array):
        error = ctypes.get_last_error()
        raise OSError(error, f"SendInput sent {sent}/{len(array)} keyboard events")


def _vk_for_key(key: str) -> int:
    normalized = key.replace("_", "").replace("-", "").upper()
    if normalized in _KEYS:
        return _KEYS[normalized]
    if len(key) == 1:
        vk = _user32().VkKeyScanW(key) & 0xFF
        if vk:
            return int(vk)
    raise ValueError(f"Unsupported key: {key}")


def key_down(key: str) -> None:
    _user32().keybd_event(_vk_for_key(key), 0, 0, 0)


def key_up(key: str) -> None:
    _user32().keybd_event(_vk_for_key(key), 0, KEYEVENTF_KEYUP, 0)


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


def type_text_unicode(text: str) -> None:
    for char in text:
        code = ord(char)
        _send_keyboard_inputs(
            (
                KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0),
                KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0),
            ),
        )


def type_text(text: str) -> None:
    if not text:
        return
    try:
        paste_text(text)
    except OSError:
        type_text_unicode(text)


def paste_text(text: str, *, restore_clipboard: bool = True) -> None:
    had_text, old_text = _get_clipboard_text()
    _set_clipboard_text(text)
    # Give the foreground control a short moment after focus/click transitions.
    time.sleep(0.03)
    press_keys(("ctrl", "v"))
    time.sleep(0.08)
    if not restore_clipboard:
        return
    if had_text:
        _set_clipboard_text(old_text or "")


def _get_clipboard_text() -> tuple[bool, str | None]:
    user32 = _user32()
    if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        return (False, None)
    _open_clipboard()
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return (True, "")
        pointer = _kernel32().GlobalLock(handle)
        if not pointer:
            return (True, "")
        try:
            return (True, ctypes.wstring_at(pointer))
        finally:
            _kernel32().GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _set_clipboard_text(text: str) -> None:
    data = ctypes.create_unicode_buffer(str(text) + "\0")
    size = ctypes.sizeof(data)
    handle = _kernel32().GlobalAlloc(GMEM_MOVEABLE, size)
    if not handle:
        error = ctypes.get_last_error()
        raise OSError(error, "GlobalAlloc failed for clipboard text")
    pointer = _kernel32().GlobalLock(handle)
    if not pointer:
        error = ctypes.get_last_error()
        _kernel32().GlobalFree(handle)
        raise OSError(error, "GlobalLock failed for clipboard text")
    try:
        ctypes.memmove(pointer, ctypes.addressof(data), size)
    finally:
        _kernel32().GlobalUnlock(handle)

    user32 = _user32()
    _open_clipboard()
    try:
        if not user32.EmptyClipboard():
            error = ctypes.get_last_error()
            _kernel32().GlobalFree(handle)
            raise OSError(error, "EmptyClipboard failed")
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            error = ctypes.get_last_error()
            _kernel32().GlobalFree(handle)
            raise OSError(error, "SetClipboardData failed")
        handle = None
    finally:
        user32.CloseClipboard()
        if handle:
            _kernel32().GlobalFree(handle)


def _open_clipboard() -> None:
    user32 = _user32()
    for _ in range(8):
        if user32.OpenClipboard(None):
            return
        time.sleep(0.02)
    error = ctypes.get_last_error()
    raise OSError(error, "OpenClipboard failed")
