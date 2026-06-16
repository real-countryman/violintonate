

def hz_to_notes(hz: float):
    """Converts hertz to notes"""
    return 12 * math.log2(hz/440)

