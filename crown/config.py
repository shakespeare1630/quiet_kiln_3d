from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    drafts: tuple[str, ...] = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol")
    planner: str = "gpt-6.1-sol"
    reviewer: str = "gpt-6.1-sol"
    selector: str = "gpt-6.1-sol"
    final_models: tuple[str, ...] = ("gpt-6-astra", "gpt-6.1-sol")
    visual_critique: bool = True
    final_refinements: int = 2
    repairs: int = 1
    max_credits: int = 4_900_000
    max_calls: int = 60
    # Subscription inference may not support a provider-side output limit.
    # These are reservations, not claims about an enforced token ceiling.
    generation_reserve_tokens: int = 32_768
    decision_reserve_tokens: int = 4_096
    image_reserve_tokens: int = 16_384
    framing_reserve_tokens: int = 2_048
    send_output_limit: bool = False
    selection_gap: float = 0.7
    refinement_consensus_gap: float = 0.35
    min_consistent_angles: int = 3

    def __post_init__(self):
        if not 1 <= len(self.drafts) <= 6:
            raise ValueError("Use one to six independent drafts")
        if not self.final_models:
            raise ValueError("Specify a refinement model")
        if not 0 < self.max_credits <= 5_000_000 or not 0 < self.max_calls <= 60:
            raise ValueError("Configuration exceeds the competition budget")
        if not 0 <= self.final_refinements <= 5 or not 0 <= self.repairs <= 2:
            raise ValueError("Invalid refinement or repair count")
        if not 0 < self.image_reserve_tokens <= 32_768 or not 0 < self.framing_reserve_tokens <= 32_768:
            raise ValueError("Invalid input token reservation")


DEFAULT = Config()
