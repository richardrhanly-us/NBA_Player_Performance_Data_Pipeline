"""
Provider-facing exceptions.

Deliberately minimal: only one distinguishable failure mode currently
exists anywhere in this codebase's NBA access (a request exhausting its
retry policy -- see training.data.nba_client.NbaApiError), so only one
concrete exception is defined below. ProviderError is the base class
callers can catch broadly; add a more specific subclass (e.g. for a
malformed payload, or a genuinely-missing entity a provider can detect)
only once a real, distinguishable failure mode exists to justify it --
not speculatively now.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all basketball-data-provider failures."""


class ProviderUnavailableError(ProviderError):
    """
    The provider could not fulfill the request after its own retry
    policy was exhausted -- a transient/availability failure, not a
    malformed-request or programmer error. Raised by NBAApiProvider when
    training.data.nba_client.NbaApiError propagates out of a fetch.
    """
