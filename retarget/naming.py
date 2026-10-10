"""Name tables: how existing rigs (MMD, VRM, ARKit-native, Mixamo, ...) spell
the things the canonical space talks about.

Two kinds of table:

* BONE_ALIASES    semantic bone -> candidate node names (first match wins)
* MORPH_DRIVERS   known morph name -> how to drive it from ARKit inputs

A driver is {"inputs": [...], "combine": op, "gain": g}; at runtime the
morph influence is clamp(g * combine(inputs)).  `combine`:
    max   any input                      min   all inputs together
    mean  average of inputs              diff  first input minus the rest (a wink = blink one side only)

Left/Right in MMD names are the *character's* sides, same as ARKit.
`ウィンク` closes the character's left eye, `ウィンク右` the right one.
"""
from __future__ import annotations

BONE_ALIASES: dict[str, tuple[str, ...]] = {
    'head': ('頭', 'Head', 'head', 'HEAD', 'mixamorig:Head', 'J_Bip_C_Head', 'Bip01 Head', 'CC_Base_Head', 'DEF-head'),
    'neck': ('首', 'Neck', 'neck', 'mixamorig:Neck', 'J_Bip_C_Neck', 'Bip01 Neck', 'CC_Base_NeckTwist01', 'DEF-neck'),
    'leftEye': ('左目', 'LeftEye', 'leftEye', 'Eye_L', 'eye_L', 'eye.L', 'l_eye', 'L_eye', 'mixamorig:LeftEye',
                'J_Adj_L_FaceEye', 'CC_Base_L_Eye', 'DEF-eye.L'),
    'rightEye': ('右目', 'RightEye', 'rightEye', 'Eye_R', 'eye_R', 'eye.R', 'r_eye', 'R_eye', 'mixamorig:RightEye',
                 'J_Adj_R_FaceEye', 'CC_Base_R_Eye', 'DEF-eye.R'),
    'jaw': ('あご', '顎', 'Jaw', 'jaw', 'CC_Base_JawRoot', 'DEF-jaw'),
}

_L = ('eyeBlinkLeft',)
_R = ('eyeBlinkRight',)

# MMD (Japanese) morph names in common use.  Spelling variants (ウィンク / ウインク,
# full-width / half-width katakana) are normalised by `normalize_morph_name`.
MMD_MORPH_DRIVERS: dict[str, dict] = {
    # --- mouth
    'あ':      {'inputs': ['jawOpen'], 'combine': 'max'},
    'あ２':    {'inputs': ['jawOpen'], 'combine': 'max', 'gain': 0.6},
    'い':      {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'う':      {'inputs': ['mouthPucker'], 'combine': 'max'},
    'え':      {'inputs': ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen'], 'combine': 'min'},
    'お':      {'inputs': ['mouthFunnel'], 'combine': 'max'},
    'ん':      {'inputs': ['mouthClose'], 'combine': 'max'},
    '▲':      {'inputs': ['mouthPucker', 'jawOpen'], 'combine': 'min'},
    '∧':      {'inputs': ['mouthRollUpper', 'mouthRollLower'], 'combine': 'mean'},
    '口角上げ': {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    '口角下げ': {'inputs': ['mouthFrownLeft', 'mouthFrownRight'], 'combine': 'mean'},
    '口横広げ': {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'にやり':  {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean', 'gain': 0.7},
    'にっこり': {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    'にこり':  {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    'ぺろっ':  {'inputs': ['tongueOut'], 'combine': 'max'},
    'てへぺろ': {'inputs': ['tongueOut'], 'combine': 'max'},
    # --- eyes
    'まばたき': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'min'},
    'ウィンク':  {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'diff'},
    'ウィンク右': {'inputs': ['eyeBlinkRight', 'eyeBlinkLeft'], 'combine': 'diff'},
    '笑い':    {'inputs': ['eyeSquintLeft', 'eyeSquintRight', 'mouthSmileLeft', 'mouthSmileRight'], 'combine': 'min'},
    'じと目':  {'inputs': ['eyeSquintLeft', 'eyeSquintRight'], 'combine': 'mean'},
    'びっくり': {'inputs': ['eyeWideLeft', 'eyeWideRight'], 'combine': 'mean'},
    'なごみ':  {'inputs': ['eyeSquintLeft', 'eyeSquintRight'], 'combine': 'mean', 'gain': 0.5},
    # --- brows
    '上':      {'inputs': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    '下':      {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    '怒り':    {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    '困る':    {'inputs': ['browInnerUp'], 'combine': 'max'},
    '真面目':  {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean', 'gain': 0.5},
    '眉頭左':  {'inputs': ['browInnerUp'], 'combine': 'max'},
    '眉頭右':  {'inputs': ['browInnerUp'], 'combine': 'max'},
    # --- cheeks
    '照れ':    {'inputs': ['cheekSquintLeft', 'cheekSquintRight'], 'combine': 'mean'},
}

# VRM 1.0 expression presets (and the VRM 0.x BlendShape preset names in parentheses)
VRM_MORPH_DRIVERS: dict[str, dict] = {
    'aa': {'inputs': ['jawOpen'], 'combine': 'max'},
    'ih': {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'ou': {'inputs': ['mouthPucker'], 'combine': 'max'},
    'ee': {'inputs': ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen'], 'combine': 'min'},
    'oh': {'inputs': ['mouthFunnel'], 'combine': 'max'},
    'blink': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'min'},
    'blinkLeft': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'diff'},
    'blinkRight': {'inputs': ['eyeBlinkRight', 'eyeBlinkLeft'], 'combine': 'diff'},
    'lookUp': {'inputs': ['eyeLookUpLeft', 'eyeLookUpRight'], 'combine': 'mean'},
    'lookDown': {'inputs': ['eyeLookDownLeft', 'eyeLookDownRight'], 'combine': 'mean'},
    'lookLeft': {'inputs': ['eyeLookOutLeft', 'eyeLookInRight'], 'combine': 'mean'},
    'lookRight': {'inputs': ['eyeLookInLeft', 'eyeLookOutRight'], 'combine': 'mean'},
    'happy': {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    'angry': {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    'sad': {'inputs': ['mouthFrownLeft', 'mouthFrownRight', 'browInnerUp'], 'combine': 'mean'},
    'surprised': {'inputs': ['eyeWideLeft', 'eyeWideRight', 'jawOpen'], 'combine': 'mean'},
}

# English / VRoid-style morph names seen on VRM and game-engine exports (lower-cased keys).
# Side suffixes _l/_r / _left/_right are stripped before lookup and re-applied to the inputs.
ENGLISH_MORPH_DRIVERS: dict[str, dict] = {
    # mouth / vowels
    'lip_a': {'inputs': ['jawOpen'], 'combine': 'max'}, 'mth_a': {'inputs': ['jawOpen'], 'combine': 'max'},
    'lip_i': {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'mth_i': {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'lip_u': {'inputs': ['mouthPucker'], 'combine': 'max'}, 'mth_u': {'inputs': ['mouthPucker'], 'combine': 'max'},
    'lip_e': {'inputs': ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen'], 'combine': 'min'},
    'mth_e': {'inputs': ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen'], 'combine': 'min'},
    'lip_o': {'inputs': ['mouthFunnel'], 'combine': 'max'}, 'mth_o': {'inputs': ['mouthFunnel'], 'combine': 'max'},
    'lip_w': {'inputs': ['mouthPucker'], 'combine': 'max'},
    'lip_up': {'inputs': ['mouthUpperUpLeft', 'mouthUpperUpRight'], 'combine': 'mean'},
    'lip_down': {'inputs': ['mouthLowerDownLeft', 'mouthLowerDownRight'], 'combine': 'mean'},
    'mouse_open': {'inputs': ['jawOpen'], 'combine': 'max'}, 'mouth_open': {'inputs': ['jawOpen'], 'combine': 'max'},
    'mouth_wide': {'inputs': ['mouthStretchLeft', 'mouthStretchRight'], 'combine': 'mean'},
    'mouth_short': {'inputs': ['mouthPucker'], 'combine': 'max', 'gain': 0.6},
    'mouth_smile': {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    'mth_fun': {'inputs': ['mouthSmileLeft', 'mouthSmileRight'], 'combine': 'mean'},
    'mth_joy': {'inputs': ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen'], 'combine': 'min'},
    'mth_sorrow': {'inputs': ['mouthFrownLeft', 'mouthFrownRight'], 'combine': 'mean'},
    'mth_angry': {'inputs': ['mouthFrownLeft', 'mouthFrownRight', 'mouthPressLeft', 'mouthPressRight'], 'combine': 'mean'},
    'mth_surprised': {'inputs': ['jawOpen', 'mouthFunnel'], 'combine': 'min'},
    'mth_up': {'inputs': ['mouthUpperUpLeft', 'mouthUpperUpRight'], 'combine': 'mean'},
    'mth_down': {'inputs': ['mouthLowerDownLeft', 'mouthLowerDownRight'], 'combine': 'mean'},
    # eyes
    'blink': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'min'},
    'eye_close': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'min'},
    'eye_close_l': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'diff'},
    'eye_close_r': {'inputs': ['eyeBlinkRight', 'eyeBlinkLeft'], 'combine': 'diff'},
    'blink_l': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'diff'},
    'blink_r': {'inputs': ['eyeBlinkRight', 'eyeBlinkLeft'], 'combine': 'diff'},
    'wink_l': {'inputs': ['eyeBlinkLeft', 'eyeBlinkRight'], 'combine': 'diff'},
    'wink_r': {'inputs': ['eyeBlinkRight', 'eyeBlinkLeft'], 'combine': 'diff'},
    'eye_open': {'inputs': ['eyeWideLeft', 'eyeWideRight'], 'combine': 'mean'},
    'eye_wide': {'inputs': ['eyeWideLeft', 'eyeWideRight'], 'combine': 'mean'},
    'eye_surprised': {'inputs': ['eyeWideLeft', 'eyeWideRight'], 'combine': 'mean'},
    'eye_small': {'inputs': ['eyeSquintLeft', 'eyeSquintRight'], 'combine': 'mean'},
    'eye_zitome': {'inputs': ['eyeSquintLeft', 'eyeSquintRight'], 'combine': 'mean'},
    'eye_smile': {'inputs': ['eyeSquintLeft', 'eyeSquintRight', 'mouthSmileLeft', 'mouthSmileRight'], 'combine': 'min'},
    'eye_joy': {'inputs': ['eyeSquintLeft', 'eyeSquintRight', 'mouthSmileLeft', 'mouthSmileRight'], 'combine': 'min'},
    'eye_angry': {'inputs': ['browDownLeft', 'browDownRight', 'eyeSquintLeft', 'eyeSquintRight'], 'combine': 'min'},
    # brows
    'eye_brow_up': {'inputs': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    'eye_brow_up_l': {'inputs': ['browOuterUpLeft', 'browInnerUp'], 'combine': 'mean'},
    'eye_brow_up_r': {'inputs': ['browOuterUpRight', 'browInnerUp'], 'combine': 'mean'},
    'eye_brow_down': {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    'eye_brow_down_l': {'inputs': ['browDownLeft'], 'combine': 'max'},
    'eye_brow_down_r': {'inputs': ['browDownRight'], 'combine': 'max'},
    'brow_up': {'inputs': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    'brow_down': {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    'brw_angry': {'inputs': ['browDownLeft', 'browDownRight'], 'combine': 'mean'},
    'brw_fun': {'inputs': ['browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    'brw_joy': {'inputs': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    'brw_sorrow': {'inputs': ['browInnerUp'], 'combine': 'max'},
    'brw_surprised': {'inputs': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'], 'combine': 'mean'},
    # cheeks
    'cheek_pop': {'inputs': ['cheekPuff'], 'combine': 'max'},
    'cheek_puff': {'inputs': ['cheekPuff'], 'combine': 'max'},
}

_KATAKANA_VARIANTS = {
    'ウインク': 'ウィンク',
    'ｳｨﾝｸ': 'ウィンク',
    'ｳｲﾝｸ': 'ウィンク',
    '２': '2',
    '2': '２',
}


def normalize_morph_name(name: str) -> str:
    """Fold common spelling variants so the tables need one entry per morph."""
    n = name.strip()
    n = n.replace('ウインク', 'ウィンク').replace('ｳｨﾝｸ', 'ウィンク').replace('ｳｲﾝｸ', 'ウィンク')
    n = n.replace('2', '２')
    return n


def _english_driver(name: str) -> dict | None:
    """ENGLISH_MORPH_DRIVERS lookup with VRoid prefixes (Fcl_) stripped and lower-casing."""
    key = name.lower()
    if key.startswith('fcl_'):
        key = key[4:]
    if key in ENGLISH_MORPH_DRIVERS:
        return dict(ENGLISH_MORPH_DRIVERS[key])
    key2 = key.replace('_left', '_l').replace('_right', '_r')
    if key2 in ENGLISH_MORPH_DRIVERS:
        return dict(ENGLISH_MORPH_DRIVERS[key2])
    return None


def lookup_morph_driver(name: str) -> dict | None:
    n = normalize_morph_name(name)
    if n in MMD_MORPH_DRIVERS:
        return dict(MMD_MORPH_DRIVERS[n])
    if n in VRM_MORPH_DRIVERS:
        return dict(VRM_MORPH_DRIVERS[n])
    eng = _english_driver(n)
    if eng:
        return eng
    # ARKit-native models: the morph *is* the input.  Accept common suffix styles.
    base = n.replace('_L', 'Left').replace('_R', 'Right')
    from .canonical import ARKIT_52
    if base in ARKIT_52:
        return {'inputs': [base], 'combine': 'max'}
    lowered = {a.lower(): a for a in ARKIT_52}
    if base.lower() in lowered:
        return {'inputs': [lowered[base.lower()]], 'combine': 'max'}
    # sided variants of ARKit shapes that are single in ARKit (ICT: browInnerUp_L/_R, cheekPuff_L/_R)
    for suffix in ('Left', 'Right'):
        if base.endswith(suffix) and base[:-len(suffix)] in ARKIT_52:
            return {'inputs': [base[:-len(suffix)]], 'combine': 'max'}
    return None


def resolve_bone(semantic: str, node_names: list[str]) -> str | None:
    aliases = BONE_ALIASES.get(semantic, ())
    for alias in aliases:
        if alias in node_names:
            return alias
    lowered = {n.lower(): n for n in node_names}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None
