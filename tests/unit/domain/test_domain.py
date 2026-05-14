"""Pure domain tests: draws, hands, scoring, deck."""

from __future__ import annotations

import random

import pytest

from app.domain.actions import apply_draw
from app.domain.deck import CardDeck
from app.domain.errors import IllegalDrawError
from app.domain.hand import Hand
from app.domain.scoring import empty_hands_for_players, winners_for_round
from app.settings import CARD_CONFIG


# --- apply_draw / actions ---


def test_apply_draw_appends_and_leaves_other_players_unchanged() -> None:
    hands = {"north": Hand((3,)), "south": Hand((10,))}
    next_hands = apply_draw(hands, "north", 4)
    assert next_hands["north"].cards == (3, 4)
    assert next_hands["south"].cards == (10,)
    assert hands["north"].cards == (3,)


def test_apply_draw_does_not_mutate_original_mapping() -> None:
    hands: dict[str, Hand] = {"only": Hand()}
    _ = apply_draw(hands, "only", 7)
    assert hands["only"].cards == ()


def test_unknown_player_raises() -> None:
    with pytest.raises(IllegalDrawError, match="Unknown"):
        apply_draw({"a": Hand()}, "missing", 5)


@pytest.mark.parametrize("bad_value", [1, 12, 0, -3])
def test_invalid_card_value_raises(bad_value: int) -> None:
    with pytest.raises(IllegalDrawError, match="card_value"):
        apply_draw({"a": Hand()}, "a", bad_value)


def test_cannot_draw_when_at_max_draws_from_config() -> None:
    cap = CARD_CONFIG.max_cards_in_hand()
    hands = {"p": Hand((4,) * cap)}
    with pytest.raises(IllegalDrawError, match="cannot draw"):
        apply_draw(hands, "p", 2)


def test_cannot_draw_when_already_bust() -> None:
    hands = {"p": Hand((10, 10, 5))}
    with pytest.raises(IllegalDrawError, match="cannot draw"):
        apply_draw(hands, "p", 2)


def test_allowed_boundary_card_values() -> None:
    hands = {"p": Hand()}
    assert apply_draw(hands, "p", 2)["p"].total == 2
    hands = {"p": Hand()}
    assert apply_draw(hands, "p", 11)["p"].total == 11


# --- Hand ---


def test_empty_hand_totals_zero_and_is_not_bust() -> None:
    hand = Hand()
    assert hand.total == 0
    assert hand.card_count == 0
    assert hand.is_bust() is False


def test_total_sums_cards() -> None:
    hand = Hand((5, 6, 7))
    assert hand.total == 18
    assert hand.card_count == 3


@pytest.mark.parametrize(
    ("cards", "expected_bust"),
    [
        ((10, 9), False),
        ((10, 10), False),
        ((10, 11), False),
        ((10, 9, 3), True),
        ((7, 7, 8), True),
    ],
)
def test_is_bust(cards: tuple[int, ...], expected_bust: bool) -> None:
    assert Hand(cards).is_bust() is expected_bust


def test_may_take_another_card_respects_max_and_bust() -> None:
    assert Hand().may_take_another_card(3) is True
    assert Hand((10, 9)).may_take_another_card(3) is True
    assert Hand((10, 9, 2)).may_take_another_card(3) is False
    assert Hand((10, 10, 2)).may_take_another_card(3) is False
    assert Hand((11, 10, 2)).may_take_another_card(3) is False
    assert Hand((11, 10)).may_take_another_card(3) is True


def test_with_card_returns_new_hand() -> None:
    first = Hand((4,))
    second = first.with_card(5)
    assert first.cards == (4,)
    assert second.cards == (4, 5)


def test_custom_max_cards_total_limit() -> None:
    hand = Hand((3, 3))
    assert hand.may_take_another_card(max_cards_total=2) is False


# --- scoring ---


def test_empty_hands_for_players_preserves_order_of_ids() -> None:
    hands = empty_hands_for_players(["z", "a", "m"])
    assert list(hands.keys()) == ["z", "a", "m"]
    assert all(hand.cards == () for hand in hands.values())


def test_single_winner_highest_non_bust_total() -> None:
    hands = {
        "alice": Hand((10, 7)),
        "bob": Hand((9, 9)),
        "cara": Hand((5, 5, 5)),
    }
    assert winners_for_round(hands) == ["bob"]


def test_tie_returns_sorted_player_ids() -> None:
    hands = {
        "zeke": Hand((10, 7)),
        "amy": Hand((9, 8)),
        "bea": Hand((6, 5, 6)),
    }
    assert winners_for_round(hands) == ["amy", "bea", "zeke"]


def test_busted_players_are_excluded() -> None:
    hands = {
        "low": Hand((10, 10, 2)),
        "high": Hand((10, 9)),
    }
    assert winners_for_round(hands) == ["high"]


def test_all_bust_returns_empty_list() -> None:
    hands = {
        "one": Hand((10, 10, 5)),
        "two": Hand((11, 10, 2)),
    }
    assert winners_for_round(hands) == []


# --- CardDeck ---


def test_draw_value_always_in_range_for_seeded_deck() -> None:
    deck = CardDeck(random.Random(42))
    for _ in range(200):
        value = deck.draw()
        assert 2 <= value <= 11


def test_seeded_deck_is_deterministic() -> None:
    first_deck = CardDeck(random.Random(123))
    second_deck = CardDeck(random.Random(123))
    for _ in range(50):
        assert first_deck.draw() == second_deck.draw()


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_known_seed_first_draw(seed: int) -> None:
    deck = CardDeck(random.Random(seed))
    first = deck.draw()
    assert 2 <= first <= 11
