"""
PS1 Mask Bit Flag Documentation (REAL PS1 DEFINITIONS)

PS1 uses a multi-bit binary mask system where each bit represents a specific condition.
These are the actual bit definitions from PS1 FITS headers:

Bit 0 (1):     DETECTOR - Detector issue
Bit 1 (2):     FLAT - Flat field issue
Bit 2 (4):     DARK - Dark current issue
Bit 3 (8):     BLANK - Blank/missing data
Bit 4 (16):    CTE - Charge transfer efficiency issue
Bit 5 (32):    SAT - Saturated pixel
Bit 6 (64):    LOW - Low signal/noise
Bit 7 (128):   SUSPECT - Suspect pixel
Bit 8 (128):   BURNTOOL - Burnt tool (same value as SUSPECT - check!)
Bit 9 (256):   CR - Cosmic ray
Bit 10 (512):  SPIKE - Spike/glitch
Bit 11 (1024): GHOST - Ghost/reflection
Bit 12 (2048): STREAK - Streak or linear feature
Bit 13 (4096): STARCORE - Star core
Bit 14 (8192): CONV.BAD - Convolution bad
Bit 15 (16384): CONV.POOR - Convolution poor
Special values:
- MASK.VALUE (8575): Combined mask value
- MARK.VALUE (32768): Mark value

Each pixel's mask value is the sum of all applicable flags.
"""

# PS1 mask bit definitions (REAL PS1 VALUES)
PS1_MASK_BITS = {
    "DETECTOR": 1,  # Bit 0
    "FLAT": 2,  # Bit 1
    "DARK": 4,  # Bit 2
    "BLANK": 8,  # Bit 3
    "CTE": 16,  # Bit 4
    "SAT": 32,  # Bit 5
    "LOW": 64,  # Bit 6
    "SUSPECT": 128,  # Bit 7
    "BURNTOOL": 128,  # Bit 7
    "CR": 256,  # Bit 8
    "SPIKE": 512,  # Bit 9
    "GHOST": 1024,  # Bit 10
    "STREAK": 2048,  # Bit 11
    "STARCORE": 4096,  # Bit 12
    "CONV.BAD": 8192,  # Bit 13
    "CONV.POOR": 16384,  # Bit 14
}

# Special PS1 mask values
PS1_SPECIAL_VALUES = {
    "MASK.VALUE": 8575,  # Combined mask value
    "MARK.VALUE": 32768,  # Mark value
}


def decode_ps1_mask(mask_value):
    """Decode PS1 mask value into individual flags."""
    flags = []

    # Check for special values first
    if mask_value == PS1_SPECIAL_VALUES["MASK.VALUE"]:
        flags.append("MASK.VALUE")
    elif mask_value == PS1_SPECIAL_VALUES["MARK.VALUE"]:
        flags.append("MARK.VALUE")
    else:
        # Check individual bit flags
        for flag_name, bit_value in PS1_MASK_BITS.items():
            if mask_value & bit_value:
                flags.append(flag_name)

    return flags


def encode_ps1_mask(flags):
    """Encode list of flags into PS1 mask value."""
    mask_value = 0
    for flag in flags:
        if flag in PS1_MASK_BITS:
            mask_value |= PS1_MASK_BITS[flag]
        elif flag in PS1_SPECIAL_VALUES:
            return PS1_SPECIAL_VALUES[flag]
    return mask_value
