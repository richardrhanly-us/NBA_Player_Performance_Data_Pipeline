"""
Basketball-data provider boundary: our own canonical contract for the
basketball data this project needs, decoupled from any one external
source's response shape.

Why this exists (Step 6, commercial-readiness architecture)
-------------------------------------------------------------------------
Before this package existed, feature engineering, historical collection,
and training all sat directly on nba_api-specific objects and column
names. That made the NBA source a permanent, implicit part of this
project's data contract rather than a swappable implementation detail --
a real problem if this project is ever backed by a licensed commercial
provider (e.g. SportsDataIO, Sportradar) instead of stats.nba.com.

The rule going forward:

    external provider -> provider adapter -> canonical internal contract
        -> collection/normalization -> feature engineering -> training/inference

instead of:

    nba_api-specific objects -> feature engineering / training / app

Currently active provider
-------------------------------------------------------------------------
The NBA development provider (providers/nba_api_provider.py::NBAApiProvider)
is the only implementation, and it is what this project runs against
today. It is explicitly a DEVELOPMENT provider, not a permanent
commitment to stats.nba.com as the data source -- see get_basketball_provider()
in provider.py.

No commercial provider is implemented or integrated in Step 6. This
package intentionally contains no SportsDataIOProvider/SportradarProvider/
"CommercialProvider" placeholder -- that would be speculative,
unauthorized architecture. A real commercial implementation, when one is
selected, is added later by implementing BasketballDataProvider (see
provider.py) and a normalization path into models.py's canonical types;
nothing else in this codebase should need to change.

What lives here
-------------------------------------------------------------------------
- models.py         canonical domain objects (Player, PlayerGameLog) --
                     OUR field names/types, not a vendor's.
- provider.py        the BasketballDataProvider Protocol every source
                     (current or future) must satisfy, plus
                     get_basketball_provider() to obtain the active one.
- normalization.py    pure, provider-agnostic mapping between an
                     already-normalized tabular payload and our
                     canonical records. Column-order-independent.
- errors.py          the (small) provider-facing exception hierarchy.
- providers/         concrete provider implementations.
    nba_api_provider.py   NBAApiProvider -- the current, active,
                          NBA-backed development provider.

What is explicitly OUT of scope here
-------------------------------------------------------------------------
- Market/odds data (sportsbook lines, closing lines, edge calculations)
  is a SEPARATE domain, not a basketball-data-provider concern. It is
  not represented anywhere in this package, and existing sportsbook code
  is untouched by this package's introduction.
- Capabilities the project does not yet consume (injuries, lineups,
  projected minutes, schedule beyond what today's collection path
  needs) are NOT modeled here. Add them in a later step, when a real
  caller needs them -- not speculatively now.
- The live Streamlit app's own direct NBA calls (src/shared_app.py:
  CommonPlayerInfo, a second PlayerGameLog call, ScoreboardV2, the live
  boxscore endpoint) are NOT migrated to this provider boundary in Step
  6. Only the historical collection path
  (training/data/collect_gamelogs.py) is migrated. See that module's
  docstring, and the Step 6 report, for why.
"""
