from typing import Tuple


def clmbr_meds_code_candidates(code: str) -> Tuple[str, ...]:
    candidates = [code]
    if "//" in code:
        parts = code.split("//")
        if len(parts) >= 2 and parts[0] and parts[1]:
            candidates.append(f"{parts[0]}/{parts[1]}")
    return tuple(dict.fromkeys(candidates))


def clmbr_meds_raw_variants(code: str) -> Tuple[str, ...]:
    variants = [code]
    if "/" in code and "//" not in code:
        vocab, concept = code.split("/", 1)
        if vocab and concept:
            variants.extend(
                [
                    f"{vocab}//{concept}",
                    f"{vocab}//{concept}//start",
                    f"{vocab}//{concept}//end",
                ]
            )
    return tuple(dict.fromkeys(variants))
