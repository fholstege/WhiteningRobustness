"""Shared average-loss ridge grid from the manuscript's evaluation protocol."""

RIDGE_LAMBDAS = [scale * 10.0**power for power in range(-5, 7) for scale in (1, 5)]
