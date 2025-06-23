def brisque_score_to_quality_label(score: int) -> str:
    if score <= 0:
        return "a-perfect"
    elif score <= 10:
        return "b-excellent"
    elif score <= 20:
        return "c-very good"
    elif score <= 30:
        return "d-good"
    elif score <= 40:
        return "e-decent"
    elif score <= 50:
        return "f-fair"
    elif score <= 60:
        return "g-mediocre"
    elif score <= 70:
        return "h-poor"
    elif score <= 80:
        return "i-very poor"
    elif score <= 90:
        return "j-awful"
    else:
        return "x-rejected"


def prompt_score_to_quality_label(quality_label: str) -> str:
    if "low" in quality_label:
        return "low_quality"
    elif "standard" in quality_label:
        return "standard_quality"
    elif "high" in quality_label:
        return "high_quality"
    elif "premium" in quality_label:
        return "premium_quality"
    else:
        return "unclassified"
