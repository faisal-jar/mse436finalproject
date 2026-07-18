import math
from . import config


def base_expected_return(prob_by_class, tier_revenue):
    """Expected revenue if the song fully realises its modelled potential."""
    return sum(prob_by_class[c] * tier_revenue[c] for c in config.CLASS_ORDER)


def realised_fraction(budget, floor=config.PROMO_FLOOR, scale=config.PROMO_SCALE):
    """Share of potential revenue captured at this spend (diminishing returns)."""
    return floor + (1 - floor) * (1 - math.exp(-budget / scale))


def recommend_action(p_at_least_B, commit_threshold):
    """Map success probability + analyst threshold to a playbook tier id."""
    if p_at_least_B >= commit_threshold * 1.5:
        return 3
    if p_at_least_B >= commit_threshold:
        return 2
    return 1


def expected_value(prob_by_class, committed_budget, tier_revenue,
                   floor=config.PROMO_FLOOR, scale=config.PROMO_SCALE):
    """Expected economics of committing `committed_budget` behind this track."""
    base = base_expected_return(prob_by_class, tier_revenue)
    real = realised_fraction(committed_budget, floor, scale)
    exp_return = base * real
    net = exp_return - committed_budget
    roi = (net / committed_budget) if committed_budget > 0 else float("nan")
    return {"expected_return": exp_return, "expected_net": net,
            "expected_roi": roi, "base_return": base, "realised": real}


def optimal_budget(prob_by_class, tier_revenue,
                   floor=config.PROMO_FLOOR, scale=config.PROMO_SCALE):
    """
    Budget that maximises expected NET value. Closed form from setting
    d/db [ base*(floor+(1-floor)(1-e^(-b/scale))) - b ] = 0:
        b* = scale * ln( base*(1-floor) / scale )
    If base*(1-floor) <= scale the marginal first dollar already loses money,
    so the optimum is $0 (don't fund a paid campaign).
    """
    base = base_expected_return(prob_by_class, tier_revenue)
    denom = base * (1 - floor)
    if denom <= scale:
        return 0.0
    return max(0.0, scale * math.log(denom / scale))


def build_plan(prediction, *,
               commit_threshold=config.DEFAULT_COMMIT_THRESHOLD,
               playbook=None, tier_revenue=None, budget_override=None):
    """Assemble the full decision object the UI renders."""
    playbook = playbook or config.DEFAULT_PLAYBOOK
    tier_revenue = tier_revenue or config.DEFAULT_TIER_REVENUE
    prob_by_class = prediction["prob_by_class"]
    p_B = prediction["p_at_least_B"]

    rec_class = recommend_action(p_B, commit_threshold)
    plan = playbook[rec_class]
    budget = budget_override if budget_override is not None else plan["budget"]

    econ = expected_value(prob_by_class, budget, tier_revenue)
    opt = optimal_budget(prob_by_class, tier_revenue)

    channel_dollars = {ch: round(frac * budget)
                       for ch, frac in plan["channel_mix"].items()}

    if econ["expected_net"] > 0 and p_B >= commit_threshold:
        verdict = "COMMIT"
    elif econ["expected_net"] > 0:
        verdict = "MARGINAL - below your confidence bar but positive EV"
    else:
        verdict = "HOLD / REVISE - expected value is negative"

    return {
        "recommended_tier": rec_class,
        "recommended_tier_meta": config.TIERS[rec_class],
        "action": plan["action"], "note": plan["note"],
        "committed_budget": budget,
        "channel_dollars": channel_dollars, "channel_mix": plan["channel_mix"],
        "economics": econ,
        "optimal_budget": opt,
        "p_at_least_B": p_B, "commit_threshold": commit_threshold,
        "verdict": verdict,
    }
